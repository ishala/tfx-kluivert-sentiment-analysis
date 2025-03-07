import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
import os
from tfx.components.trainer.fn_args_utils import FnArgs
from collections import namedtuple

# Named tuple untuk hasil tuning
TunerFnResult = namedtuple('TunerFnResult', ['tuner', 'fit_kwargs'])

LABEL_KEY = "polarity"
FEATURE_KEY = "comment"
MAX_SEQUENCE_LENGTH = 200
comb_embedding = [32, 64, 128]
comb_lstm = [64, 128]
comb_dropout = [0.3, 0.4]
comb_dense = [32, 64]


def transformed_name(key):
    return key + "_xf"


def gzip_reader_fn(filenames):
    """Load compressed TFRecord dataset"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def get_vocab_size(tf_transform_output):
    """Membaca ukuran vocabulary dari transform graph."""
    vocab_path = os.path.join(
        tf_transform_output.transform_savedmodel_dir,
        "assets",
        "vocab")

    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}")

    with open(vocab_path, "r") as f:
        vocab_list = f.readlines()
    return len(vocab_list) + 5  # Menyesuaikan dengan num_oov_buckets


def count_samples(file_patterns, tf_transform_output):
    dataset = input_fn(
        file_patterns,
        tf_transform_output,
        num_epochs=1,
        batch_size=1)
    sample_count = sum(1 for _ in dataset)
    if sample_count == 0:
        raise ValueError(
            "Dataset is empty. Check the file patterns and data source.")
    return sample_count


def input_fn(
        file_pattern,
        tf_transform_output,
        num_epochs,
        batch_size=64) -> tf.data.Dataset:
    """Membaca data TFRecord dan membuat batch"""
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY)
    )

    # Cek apakah dataset kosong
    if not any(dataset):
        raise ValueError(
            "Dataset is empty. Check the file pattern and data source.")

    return dataset


def model_builder(hp, tf_transform_output):
    """Membangun model dengan hyperparameter yang akan dituning."""

    embedding_dim = hp.Choice('embedding_dim', comb_embedding)
    lstm_units = hp.Choice('lstm_units', comb_lstm)
    dropout_rate = hp.Choice('dropout_rate', comb_dropout)
    dense_units = hp.Choice('dense_units', comb_dense)

    print("Embedding dim", embedding_dim)
    print("lstm units", lstm_units)
    print("dropout rate", dropout_rate)
    print("dense units", dense_units)

    # Dapatkan vocab_size dari transform graph
    vocab_size = get_vocab_size(tf_transform_output)

    inputs = tf.keras.Input(
        shape=(
            MAX_SEQUENCE_LENGTH,
        ),
        name=transformed_name(FEATURE_KEY),
        dtype=tf.int32)
    x = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=MAX_SEQUENCE_LENGTH)(inputs)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            lstm_units,
            return_sequences=False))(x)
    x = tf.keras.layers.Dense(dense_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return model


def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """Fungsi yang dipanggil oleh pipeline untuk melakukan tuning hyperparameter."""
    tf_transform_output = tft.TFTransformOutput(
        fn_args.transform_graph_path)  # Load transform output

    # Pass `tf_transform_output` ke `model_builder`
    def model_builder_wrapper(hp):
        return model_builder(hp, tf_transform_output)

    tuner = kt.Hyperband(
        model_builder_wrapper,  # Gunakan wrapper agar bisa passing vocab_size
        objective='val_accuracy',
        max_epochs=10,
        factor=3,
        directory=fn_args.working_dir,
        project_name="sentiment_tuning"
    )

    tuner.oracle.max_trials = tuner.oracle.max_trials or 10  # Misalkan default 10

    train_set = input_fn(
        fn_args.train_files,
        tf_transform_output,
        num_epochs=10)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=10)

    # Hitung jumlah sampel
    num_train_samples = count_samples(fn_args.train_files, tf_transform_output)
    num_val_samples = count_samples(fn_args.eval_files, tf_transform_output)

    # Hitung steps per epoch
    batch_size = 64  # Sesuai dengan batch_size yang digunakan di input_fn
    steps_per_epoch = num_train_samples // batch_size
    validation_steps = num_val_samples // batch_size

    if steps_per_epoch == 0 or validation_steps == 0:
        raise ValueError(
            "Steps per epoch or validation steps cannot be zero. Check the dataset.")

    print(f"num_train_samples: {num_train_samples}")
    print(f"num_val_samples: {num_val_samples}")
    print(f"steps_per_epoch: {steps_per_epoch}")
    print(f"validation_steps: {validation_steps}")

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_set,
            "validation_data": val_set,
            "epochs": 10,
            "steps_per_epoch": steps_per_epoch,
            "validation_steps": validation_steps,
        }
    )
