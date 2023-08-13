import os
import pickle
from transformers import BertTokenizer
import sentencepiece as spm
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf


class LanguageProcessor:
    """
    This class is responsible for processing language data, including tokenizing,
    saving and loading tokenized data, and batching data with bucketing.
    """

    def __init__(self, src_lang_file_path, tgt_lang_file_path, src_vocab_file_path,
                 tgt_vocab_file_path, batch_size):
        """
        Initializes the LanguageProcessor.

        :param src_lang_file_path: Path to the Source language file.
        :param tgt_lang_file_path: Path to the Target language file.
        :param src_vocab_file_path: Path to the Source vocabulary file.
        :param tgt_vocab_file_path: Path to the Target vocabulary file.
        :param batch_size: Size of the batches for training.
        """
        self.src_file = src_lang_file_path
        self.tgt_file = tgt_lang_file_path
        self.src_vocab_file = src_vocab_file_path
        self.tgt_vocab_file = tgt_vocab_file_path
        self.batch_size = batch_size
        self.tokenizer_src = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer_tgt = None

    def tokenize(self):
        """
        Tokenizes the English and Amharic sentences.

        :return: Tokenized source and target sentences.
        """
        with open(self.src_file, 'r', encoding='utf-8') as file:
            src_sentences = file.read().split('\n')
        src_tokens = [self.tokenizer_src.encode(sentence, truncation=True, padding='max_length', max_length=128) for
                      sentence in src_sentences]

        with open(self.tgt_file, 'r', encoding='utf-8') as file:
            tgt_sentences = file.read().split('\n')

        if not os.path.exists(self.src_vocab_file + ".model"):
            spm.SentencePieceTrainer.train(input=self.src_file, model_prefix=self.src_vocab_file, vocab_size=2 ** 13)
        self.tokenizer_src = spm.SentencePieceProcessor(model_file=self.src_vocab_file + ".model")
        src_tokens = [self.tokenizer_src.encode_as_ids(sentence) for sentence in src_sentences]

        if not os.path.exists(self.tgt_vocab_file + ".model"):
            spm.SentencePieceTrainer.train(input=self.tgt_file, model_prefix=self.tgt_vocab_file, vocab_size=2 ** 13)
        self.tokenizer_tgt = spm.SentencePieceProcessor(model_file=self.tgt_vocab_file + ".model")
        tgt_tokens = [self.tokenizer_tgt.encode_as_ids(sentence) for sentence in tgt_sentences]

        return src_tokens, tgt_tokens

    def save_tokenized_data(self, src_tokens, tgt_tokens, src_file='eng_tokens.pkl', tgt_file='amh_tokens.pkl'):
        """
        Saves the tokenized data.

        :param src_tokens: Tokenized source sentences.
        :param tgt_tokens: Tokenized target sentences.
        :param src_file: File name for saving source tokens.
        :param tgt_file: File name for saving target tokens.
        """
        with open(src_file, 'wb') as f:
            pickle.dump(src_tokens, f)
        with open(tgt_file, 'wb') as f:
            pickle.dump(tgt_tokens, f)

    def load_tokenized_data(self, src_file='eng_tokens.pkl', tgt_file='amh_tokens.pkl'):
        """
        Loads the tokenized data.

        :param src_file: File name for loading source tokens.
        :param tgt_file: File name for loading target tokens.
        :return: Loaded source and target tokens.
        """
        with open(src_file, 'rb') as f:
            src_tokens = pickle.load(f)
        with open(tgt_file, 'rb') as f:
            tgt_tokens = pickle.load(f)
        return src_tokens, tgt_tokens

    def load_vocab(self):
        """
        Loads the vocabularies for both English and Amharic.

        :return: Sizes of the English and Amharic vocabularies.
        """
        if not os.path.exists(self.src_vocab_file) or not os.path.exists(self.tgt_vocab_file + ".model"):
            _, _ = self.tokenize()

        self.tokenizer_src = BertTokenizer(self.src_vocab_file)
        eng_vocab_size = self.tokenizer_src.vocab_size

        self.tokenizer_tgt = spm.SentencePieceProcessor(model_file=self.tgt_vocab_file + ".model")
        amh_vocab_size = self.tokenizer_tgt.get_piece_size()

        return eng_vocab_size, amh_vocab_size

    def get_bucketed_batches(self):
        """
       Gets batches of data with bucketing.

       :return: A dataset of bucketed batches.
       """
        if os.path.exists('eng_tokens.pkl') and os.path.exists('amh_tokens.pkl'):
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




