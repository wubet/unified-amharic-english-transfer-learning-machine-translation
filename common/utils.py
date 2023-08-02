import tensorflow as tf


# from tensorflow.keras.preprocessing.text import Tokenizer


def positional_encoding(position, d_model):
    angle_rates = 1 / tf.pow(10000, (2 * (tf.range(d_model, dtype=tf.float32) // 2)) / tf.cast(d_model, tf.float32))
    angle_rads = tf.reshape(angle_rates, (1, -1))
    angles = position * angle_rads

    sines = tf.math.sin(angles[:, 0::2])
    cosines = tf.math.cos(angles[:, 1::2])

    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = tf.expand_dims(pos_encoding, 0)

    return pos_encoding


def scaled_dot_product_attention(q, k, v, mask):
    qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    softmax_logits = qk / tf.math.sqrt(dk)
    if mask is not None:
        softmax_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(softmax_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights


def position_wise_ffn(dense_units, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(dense_units)
    ])


# function to load dataset from two sources
def load_dataset(src_file_path, tgt_file_path):
    # Load the source and target files
    src_dataset = tf.data.TextLineDataset(src_file_path)
    tgt_dataset = tf.data.TextLineDataset(tgt_file_path)

    # Zip the datasets together
    train_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

    return train_dataset


# Initialize source and target tokenizers
# These should be fit on your corpus beforehand
# src_tokenizer = Tokenizer()
# tgt_tokenizer = Tokenizer()


# def load_dataset(src_file_path, tgt_file_path, batch_size):
#     # Function to encode each sentence
#     def encode_sentence(src, tgt):
#         src = src_tokenizer.texts_to_sequences([src.numpy().decode('utf-8')])[0]
#         tgt = tgt_tokenizer.texts_to_sequences([tgt.numpy().decode('utf-8')])[0]
#         return src, tgt
#
#     # Use tf.py_function to allow the use of python function with tf.data pipeline
#     def tf_encode_sentence(src, tgt):
#         return tf.py_function(encode_sentence, [src, tgt], [tf.int64, tf.int64])
#
#     # Load the source and target files
#     src_dataset = tf.data.TextLineDataset(src_file_path)
#     tgt_dataset = tf.data.TextLineDataset(tgt_file_path)
#
#     # Zip the datasets together
#     train_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
#
#     # Map the encode_sentence function to each (src, tgt) pair in the dataset
#     train_dataset = train_dataset.map(tf_encode_sentence)
#
#     # Batch the dataset
#     train_dataset = train_dataset.batch(batch_size)
#
#     return train_dataset


# load vocabulary file
# def load_vocab(vocab_file_path):
#     with open(vocab_file_path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#     return [line.strip() for line in lines]
#
#
# # determine the size of the vocabulary file
# def vocab_size(vocab):
#     return len(vocab) + 2


# Create tokenizer from the vocab
# def create_tokenizer(vocab):
#     tokenizer = Tokenizer()
#     tokenizer.fit_on_texts(vocab)
#     return tokenizer
#
#
# def read_file_ignore_errors(file_path):
#     with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
#         return f.read()
#
#
# # Convert attention mask to 1s and 0s format and add extra dimensions to match the `hidden_states` tensor's shape
# def create_attention_mask(attention_mask):
#     attention_mask = tf.cast(attention_mask, tf.float32)
#     extended_attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
#     extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
#     return extended_attention_mask


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # add extra dimensions to match the BERT model's expected input


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_combined_masks(tgt):
    look_ahead_mask = create_look_ahead_mask(tf.shape(tgt)[1])
    dec_target_padding_mask = create_padding_mask(tgt)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return combined_mask


def create_look_ahead_attention_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    look_ahead_mask = tf.expand_dims(mask, 0)  # Now shape is (1, 441, 441)
    look_ahead_mask = tf.expand_dims(look_ahead_mask, 1)  # Now shape is (1, 1, 441, 441)
    return look_ahead_mask  # (1, 1, seq_len, seq_len)


def mse_loss(teacher_enc_output, student_enc_output):
    return tf.reduce_mean(tf.square(teacher_enc_output - student_enc_output))
