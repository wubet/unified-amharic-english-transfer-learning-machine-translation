import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer


def positional_encoding(position, d_model):
    angle_rates = 1 / tf.pow(10000, (2 * (tf.range(d_model, dtype=tf.float32) // 2)) / tf.cast(d_model, tf.float32))
    angle_rads = tf.reshape(angle_rates, (1, -1))
    angles = position * angle_rads

    sines = tf.math.sin(angles[:, 0::2])
    cosines = tf.math.cos(angles[:, 1::2])

    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = tf.expand_dims(pos_encoding, 0)

    return pos_encoding


# function to load dataset from two sources
def load_dataset(src_file_path, tgt_file_path):
    # Load the source and target files
    src_dataset = tf.data.TextLineDataset(src_file_path)
    tgt_dataset = tf.data.TextLineDataset(tgt_file_path)

    # Zip the datasets together
    train_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

    return train_dataset


from tensorflow.keras.preprocessing.text import Tokenizer

# Initialize source and target tokenizers
# These should be fit on your corpus beforehand
src_tokenizer = Tokenizer()
tgt_tokenizer = Tokenizer()


def load_dataset(src_file_path, tgt_file_path, batch_size):
    # Function to encode each sentence
    def encode_sentence(src, tgt):
        src = src_tokenizer.texts_to_sequences([src.numpy().decode('utf-8')])[0]
        tgt = tgt_tokenizer.texts_to_sequences([tgt.numpy().decode('utf-8')])[0]
        return src, tgt

    # Use tf.py_function to allow the use of python function with tf.data pipeline
    def tf_encode_sentence(src, tgt):
        return tf.py_function(encode_sentence, [src, tgt], [tf.int64, tf.int64])

    # Load the source and target files
    src_dataset = tf.data.TextLineDataset(src_file_path)
    tgt_dataset = tf.data.TextLineDataset(tgt_file_path)

    # Zip the datasets together
    train_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

    # Map the encode_sentence function to each (src, tgt) pair in the dataset
    train_dataset = train_dataset.map(tf_encode_sentence)

    # Batch the dataset
    train_dataset = train_dataset.batch(batch_size)

    return train_dataset


# load vocabulary file
def load_vocab(vocab_file_path):
    with open(vocab_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


# determine the size of the vocabulary file
def vocab_size(vocab):
    return len(vocab) + 2


# Create tokenizer from the vocab
def create_tokenizer(vocab):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(vocab)
    return tokenizer


def read_file_ignore_errors(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()
