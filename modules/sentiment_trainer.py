import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers, models  # type: ignore
import os
from tfx.components.trainer.fn_args_utils import FnArgs
from tensorflow.keras.callbacks import Callback  # type: ignore

LABEL_KEY = "polarity"
FEATURE_KEY = "comment"
MAX_SEQUENCE_LENGTH = 200


class EarlyStoppingAtThreshold(Callback):
    """Hentikan training saat val_accuracy atau accuracy mencapai threshold tertentu."""

    def __init__(self, accuracy_threshold=0.90, val_accuracy_threshold=0.85):
        super(EarlyStoppingAtThreshold, self).__init__()
        self.accuracy_threshold = accuracy_threshold
        self.val_accuracy_threshold = val_accuracy_threshold

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get("val_accuracy")
        acc = logs.get("accuracy")

        if val_acc is not None and val_acc >= self.val_accuracy_threshold:
            print(
                f"\n❗ Hentikan training: val_accuracy telah mencapai {val_acc:.4f} (threshold: {self.val_accuracy_threshold})")
            self.model.stop_training = True

        if acc is not None and acc >= self.accuracy_threshold:
            print(
                f"\n✅ Hentikan training: accuracy telah mencapai {acc:.4f} (threshold: {self.accuracy_threshold})")
            self.model.stop_training = True


def transformed_name(key):
    return key + "_xf"


def get_vocab_size(tf_transform_output):
    """Load vocab size from transform output"""
    vocab_path = os.path.join(
        tf_transform_output.transform_savedmodel_dir,
        "assets",
        "vocab")
    with open(vocab_path, "r") as f:
        vocab_list = f.readlines()
    return len(vocab_list) + 5  # Tambahkan 5 karena num_oov_buckets=5


def gzip_reader_fn(filenames):
    """Load compressed TFRecord dataset"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def input_fn(
        file_pattern,
        tf_transform_output,
        num_epochs,
        batch_size=64) -> tf.data.Dataset:
    """Get transformed features & create batches of data"""
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY)
    )

    return dataset


def count_samples(file_patterns, tf_transform_output):
    """Menghitung jumlah sampel dalam dataset TFRecord."""
    dataset = input_fn(
        file_patterns,
        tf_transform_output,
        num_epochs=1,
        batch_size=1)
    return sum(1 for _ in dataset)


def model_builder(tf_transform_output, hp):
    """Build LSTM-based model using tuned hyperparameters"""

    print("Isi hp", hp.values)
    vocab_size = get_vocab_size(tf_transform_output)
    embedding_dim = hp.get('embedding_dim', 128)
    lstm_units = hp.get('lstm_units', 128)
    dropout_rate = hp.get('dropout_rate', 0.5)
    dense_units = hp.get('dense_units', 64)

    inputs = tf.keras.Input(
        shape=(
            MAX_SEQUENCE_LENGTH,
        ),
        name=transformed_name(FEATURE_KEY),
        dtype=tf.int32)

    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=MAX_SEQUENCE_LENGTH)(inputs)
    x = layers.Bidirectional(
        layers.LSTM(
            lstm_units,
            return_sequences=False))(x)
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(3, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.summary()
    return model


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Serving function for model deployment."""
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_tf_examples_fn


def run_fn(fn_args: FnArgs) -> None:
    """Run training pipeline"""

    # Logging directory
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')

    # Callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq='batch'
    )

    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', mode='max', verbose=1, patience=5)
    mc = tf.keras.callbacks.ModelCheckpoint(
        fn_args.serving_model_dir,
        monitor='val_accuracy',
        mode='max',
        verbose=1,
        save_best_only=True)

    # Load transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Load dataset
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

    print(
        f"Total training samples: {num_train_samples}, Steps per epoch: {steps_per_epoch}")
    print(
        f"Total validation samples: {num_val_samples}, Validation steps: {validation_steps}")

    # Ambil hyperparameters dari Tuner
    hp = fn_args.hyperparameters.get(
        "values") if fn_args.hyperparameters else {}

    # Build model dengan hyperparameters yang dituning
    model = model_builder(tf_transform_output, hp)

    # Train model
    model.fit(
        train_set,
        validation_data=val_set,
        callbacks=[tensorboard_callback, es, mc],
        epochs=10,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )

    # Save model with serving function
    signatures = {
        'serving_default': _get_serve_tf_examples_fn(
            model,
            tf_transform_output).get_concrete_function(
            tf.TensorSpec(
                shape=[None],
                dtype=tf.string,
                name='examples'))}

    model.save(
        fn_args.serving_model_dir,
        save_format='tf',
        signatures=signatures)
