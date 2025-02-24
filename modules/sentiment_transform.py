import tensorflow as tf
import tensorflow_transform as tft

# Nama kolom
FEATURE_KEY = 'comment'
LABEL_KEY = 'polarity'
MAX_SEQUENCE_LENGTH = 200  # Pastikan konsisten dengan model

def transformed_name(key):
    return key + "_xf"

def preprocessing_fn(inputs):
    outputs = {}

    # Case folding (lowercasing)
    comment_dense = tf.sparse.to_dense(inputs[FEATURE_KEY], default_value="")
    comment_lower = tf.strings.lower(comment_dense)

    # Pastikan vocabulary disimpan sebagai file
    comment_indices = tft.compute_and_apply_vocabulary(
        comment_lower, num_oov_buckets=5, vocab_filename="vocab"
    )

    # Padding agar panjang tetap
    comment_sparse = tf.RaggedTensor.from_tensor(comment_indices).to_sparse()
    comment_dense_fixed = tf.sparse.to_dense(comment_sparse, default_value=0)
    comment_padded = tf.pad(
        comment_dense_fixed,
        paddings=[[0, 0], [0, MAX_SEQUENCE_LENGTH - tf.shape(comment_dense_fixed)[1]]],
        mode="CONSTANT",
        constant_values=0
    )

    comment_padded_fixed = tf.ensure_shape(comment_padded, [None, MAX_SEQUENCE_LENGTH])

    # Simpan hasil preprocessing
    outputs[transformed_name(FEATURE_KEY)] = comment_padded_fixed
    outputs[transformed_name(LABEL_KEY)] = tft.compute_and_apply_vocabulary(inputs[LABEL_KEY])

    return outputs
