from transformers import BertTokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np


class SentenceTokenizer:
    def __init__(self, src_file_path, tgt_file_path, batch_size):
        self.src_file_path = src_file_path
        self.tgt_file_path = tgt_file_path
        self.batch_size = batch_size

        # Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

    def tokenize_sentences(self, sentence):
        # Wrap encode_plus method into a tf.py_function
        sentence = tf.py_function(func=self._tokenize_sentence_py_func,
                                  inp=[sentence],
                                  Tout=(tf.int32))

        # Ensure the shape is properly set
        sentence.set_shape([None])

        return sentence

    def _tokenize_sentence_py_func(self, sentence):
        # This function will run in python
        sentence = sentence.numpy().decode('utf-8')
        tokens = self.tokenizer.encode_plus(sentence, return_tensors='np', add_special_tokens=True)

        return np.array(tokens['input_ids'][0])

    def load_and_tokenize(self):
        # Load the source and target files
        src_dataset = tf.data.TextLineDataset(self.src_file_path)
        tgt_dataset = tf.data.TextLineDataset(self.tgt_file_path)

        # Print the first sentence from source and target files
        print("First sentence in source file: ", next(iter(src_dataset)))
        print("First sentence in target file: ", next(iter(tgt_dataset)))

        # Tokenize the sentences
        src_dataset = src_dataset.map(self.tokenize_sentences)
        tgt_dataset = tgt_dataset.map(self.tokenize_sentences)

        # Zip the datasets together
        train_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

        # Function to calculate the length of an item
        def element_length_func(x, y):
            return tf.shape(x)[0]

        # Boundaries for the buckets
        boundaries = [50, 100, 150, 200, 250]  # Modify these values as needed

        # Batch sizes for each bucket
        batch_sizes = [self.batch_size] * (len(boundaries) + 1)

        # Use bucket_by_sequence_length to bucket the sentences
        train_dataset = train_dataset.apply(
            tf.data.experimental.bucket_by_sequence_length(
                element_length_func, boundaries, batch_sizes,
                padded_shapes=(([None]), ([None]))
            )
        )

        return train_dataset

    def load_vocab(self, vocab_file_path):
        try:
            with open(vocab_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            return [line.strip() for line in lines]
        except FileNotFoundError:
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_file_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading vocabulary from {vocab_file_path}: {e}")

    # determine the size of the vocabulary file
    def vocab_size(self, vocab):
        if vocab is None or len(vocab) == 0:
            raise ValueError("Vocabulary is None or empty. Cannot calculate vocabulary size.")
        return len(vocab) + 2

    # def __init__(self, src_file_path, tgt_file_path, batch_size):
    #     self.src_file_path = src_file_path
    #     self.tgt_file_path = tgt_file_path
    #     self.batch_size = batch_size
    #
    #     # Initialize the BERT tokenizer
    #     self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    #
    # def tokenize_sentences(self, sentence):
    #     # Wrap encode_plus method into a tf.py_function
    #     sentence = tf.py_function(func=self._tokenize_sentence_py_func,
    #                               inp=[sentence],
    #                               Tout=(tf.int32, tf.int32))
    #
    #     # Create a dictionary for the tensor
    #     sentence = {'input_ids': sentence[0], 'attention_mask': sentence[1]}
    #
    #     # Ensure the shape is properly set
    #     sentence['input_ids'].set_shape([None])
    #     sentence['attention_mask'].set_shape([None])
    #
    #     return sentence
    #
    # def _tokenize_sentence_py_func(self, sentence):
    #     # This function will run in python
    #     sentence = sentence.numpy().decode('utf-8')
    #     tokens = self.tokenizer.encode_plus(sentence, return_tensors='np')
    #
    #     return np.array(tokens['input_ids'][0]), np.array(tokens['attention_mask'][0])
    #
    # def load_and_tokenize(self):
    #     # Load the source and target files
    #     src_dataset = tf.data.TextLineDataset(self.src_file_path)
    #     tgt_dataset = tf.data.TextLineDataset(self.tgt_file_path)
    #
    #     # Tokenize the sentences
    #     src_dataset = src_dataset.map(self.tokenize_sentences)
    #     tgt_dataset = tgt_dataset.map(self.tokenize_sentences)
    #
    #     # Flatten the datasets and separate input_ids and attention_mask
    #     src_dataset = src_dataset.map(lambda features: (features['input_ids'], features['attention_mask']))
    #     tgt_dataset = tgt_dataset.map(lambda features: (features['input_ids'], features['attention_mask']))
    #
    #     # Zip the datasets together
    #     train_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
    #
    #     # Function to calculate the length of an item
    #     def element_length_func(x, y):
    #         return tf.shape(x)[0]
    #
    #     # Boundaries for the buckets
    #     boundaries = [50, 100, 150, 200, 250]  # Modify these values as needed
    #
    #     # Batch sizes for each bucket
    #     batch_sizes = [self.batch_size] * (len(boundaries) + 1)
    #
    #     # Use bucket_by_sequence_length to bucket the sentences
    #     train_dataset = train_dataset.apply(
    #         tf.data.experimental.bucket_by_sequence_length(
    #             element_length_func, boundaries, batch_sizes,
    #             padded_shapes=(([None], [None]), ([None], [None]))
    #         )
    #     )

        # return train_dataset

