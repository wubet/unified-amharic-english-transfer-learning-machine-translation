import os
import pickle
from transformers import BertTokenizer
import sentencepiece as spm
import tensorflow as tf
import collections
import math


class LanguageProcessor:
    """
        This class is responsible for processing language data, including tokenizing,
        saving and loading tokenized data, and batching data with bucketing.
    """

    def __init__(self, src_lang_file_path, tgt_lang_file_path, src_vocab_file_path,
                 tgt_vocab_file_path, batch_size):
        """
            Initializes the LanguageProcessor.

            :param src_lang_file_path: str, path to the Source language file.
            :param tgt_lang_file_path: str, path to the Target language file.
            :param src_vocab_file_path: str, path to the Source vocabulary file.
            :param tgt_vocab_file_path: str, path to the Target vocabulary file.
            :param batch_size: int, size of the batches for training.
        """
        self.src_file = src_lang_file_path
        self.tgt_file = tgt_lang_file_path
        self.src_vocab_file = src_vocab_file_path
        self.tgt_vocab_file = tgt_vocab_file_path
        self.batch_size = batch_size
        self.tokenizer_src = None
        self.tokenizer_tgt = None

    def tokenize(self):
        """
           Tokenizes the English and Amharic sentences.

           :return: tuple of lists, tokenized source and target sentences.
        """
        with open(self.src_file, 'r', encoding='utf-8') as file:
            src_sentences = file.read().split('\n')

        with open(self.tgt_file, 'r', encoding='utf-8') as file:
            tgt_sentences = file.read().split('\n')

        # Check if directory exists and create if not
        directory_path = "tf-record"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        # Tokenize source sentences
        if self.src_vocab_file.endswith('.en'):
            self.tokenizer_src = BertTokenizer.from_pretrained('bert-base-uncased')
            src_tokens = [self.tokenizer_src.encode(sentence, truncation=True, padding='max_length', max_length=128)
                          for sentence in src_sentences]
        elif self.src_vocab_file.endswith('.am'):
            word_counts_src = collections.Counter(word for sentence in src_sentences for word in sentence.split())
            # Determine the vocabulary size as the number of unique words
            total_words = sum(word_counts_src.values())
            # Set the vocabulary size based on the square root of the total number of words
            vocab_size_src = int(math.sqrt(total_words))
            if not os.path.exists(self.src_vocab_file + ".model"):
                spm.SentencePieceTrainer.train(input=self.src_file, model_prefix=self.src_vocab_file,
                                               vocab_size=vocab_size_src)
            self.tokenizer_src = spm.SentencePieceProcessor(model_file=self.src_vocab_file + ".model")
            src_tokens = [self.tokenizer_src.encode_as_ids(sentence) for sentence in src_sentences]

        # Tokenize target sentences
        if self.tgt_vocab_file.endswith('.en'):
            self.tokenizer_tgt = BertTokenizer.from_pretrained('bert-base-uncased')
            tgt_tokens = [self.tokenizer_tgt.encode(sentence, truncation=True, padding='max_length', max_length=128)
                          for sentence in tgt_sentences]
        elif self.tgt_vocab_file.endswith('.am'):
            word_counts_tgt = collections.Counter(word for sentence in tgt_sentences for word in sentence.split())
            total_words = sum(word_counts_tgt.values())
            # Set the vocabulary size based on the square root of the total number of words
            vocab_size_tgt = int(math.sqrt(total_words))
            if not os.path.exists(self.tgt_vocab_file + ".model"):
                spm.SentencePieceTrainer.train(input=self.tgt_file, model_prefix=self.tgt_vocab_file,
                                               vocab_size=vocab_size_tgt)
            self.tokenizer_tgt = spm.SentencePieceProcessor(model_file=self.tgt_vocab_file + ".model")
            tgt_tokens = [self.tokenizer_tgt.encode_as_ids(sentence) for sentence in tgt_sentences]

        return src_tokens, tgt_tokens

    def save_tokenized_data(self, src_tokens, tgt_tokens, src_file='src_tokens.pkl', tgt_file='tgt_tokens.pkl'):
        """
            Saves the tokenized data.

            :param src_tokens: list, tokenized source sentences.
            :param tgt_tokens: list, tokenized target sentences.
            :param src_file: str, optional, file name for saving source tokens. Default is 'src_tokens.pkl'.
            :param tgt_file: str, optional, file name for saving target tokens. Default is 'tgt_tokens.pkl'.
        """
        with open(src_file, 'wb') as f:
            pickle.dump(src_tokens, f)
        with open(tgt_file, 'wb') as f:
            pickle.dump(tgt_tokens, f)

    def load_tokenized_data(self, src_file='src_tokens.pkl', tgt_file='tgt_tokens.pkl'):
        """
            Loads the tokenized data.

            :param src_file: str, optional, file name for loading source tokens. Default is 'src_tokens.pkl'.
            :param tgt_file: str, optional, file name for loading target tokens. Default is 'tgt_tokens.pkl'.
            :return: tuple of lists, loaded source and target tokens.
        """
        with open(src_file, 'rb') as f:
            src_tokens = pickle.load(f)
        with open(tgt_file, 'rb') as f:
            tgt_tokens = pickle.load(f)
        return src_tokens, tgt_tokens

    def load_vocab(self):
        """
            Loads the vocabularies for both English and Amharic.

            :return: tuple of int, sizes of the English and Amharic vocabularies.
        """
        if not os.path.exists(self.src_vocab_file + ".model") or not os.path.exists(self.tgt_vocab_file + ".model"):
            _, _ = self.tokenize()

        if self.src_vocab_file.endswith('.en'):
            self.tokenizer_src = BertTokenizer.from_pretrained('bert-base-uncased')
            src_vocab_size = len(self.tokenizer_src.get_vocab())
        elif self.src_vocab_file.endswith('.am'):
            self.tokenizer_src = spm.SentencePieceProcessor(model_file=self.src_vocab_file + ".model")
            src_vocab_size = self.tokenizer_src.get_piece_size()

        if self.tgt_vocab_file.endswith('.en'):
            self.tokenizer_tgt = BertTokenizer.from_pretrained('bert-base-uncased')
            tgt_vocab_size = len(self.tokenizer_tgt.get_vocab())
        elif self.tgt_vocab_file.endswith('.am'):
            self.tokenizer_tgt = spm.SentencePieceProcessor(model_file=self.tgt_vocab_file + ".model")
            tgt_vocab_size = self.tokenizer_tgt.get_piece_size()

        return src_vocab_size, tgt_vocab_size
    def get_bucketed_batches(self):
        """
           Gets batches of data with bucketing.

           :return: A dataset of bucketed batches.
        """
        if os.path.exists('src_tokens.pkl') and os.path.exists('tgt_tokens.pkl'):
            src_tokens, tgt_tokens = self.load_tokenized_data()
        else:
            src_tokens, tgt_tokens = self.tokenize()
            self.save_tokenized_data(src_tokens, tgt_tokens)

        src_tokens = tf.ragged.constant(src_tokens)
        tgt_tokens = tf.ragged.constant(tgt_tokens)

        src_tokens = src_tokens.to_tensor()
        tgt_tokens = tgt_tokens.to_tensor()

        max_length = max(max(len(seq) for seq in src_tokens), max(len(seq) for seq in tgt_tokens))

        # Define the bucket boundaries and the batch sizes for each bucket
        boundaries = [10, 20, 50, 80, 110, 140, 170, 200, 230, 260, 290, 320, 350, 380, 410, max_length + 10]
        batch_sizes = [self.batch_size] * (len(boundaries) + 1)

        def element_length_fn(src, tgt):
            return tf.shape(src)[0]

        dataset = tf.data.Dataset.from_tensor_slices((src_tokens, tgt_tokens))

        # Apply bucketing
        dataset = dataset.apply(
            tf.data.experimental.bucket_by_sequence_length(
                element_length_func=element_length_fn,
                bucket_batch_sizes=batch_sizes,
                bucket_boundaries=boundaries,
                pad_to_bucket_boundary=True
            )
        )

        dataset = dataset.shuffle(10)

        return dataset




