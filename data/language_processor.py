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
        self.tokenizer = None
        self.src_file = src_lang_file_path
        self.tgt_file = tgt_lang_file_path
        self.src_vocab_file = src_vocab_file_path
        self.tgt_vocab_file = tgt_vocab_file_path
        self.batch_size = batch_size
        self.tokenizer_src = None
        self.tokenizer_tgt = None

    def tokenize(self, task):
        """
        Tokenizes the English and Amharic sentences.

        :param task: task to be performed.
        :return: tuple of lists, tokenized source and target sentences.
        """
        src_token_path = f'src_{task}_tokens.pkl'
        tgt_token_path = f'tgt_{task}_tokens.pkl'
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        if os.path.exists(src_token_path) and os.path.exists(tgt_token_path):
            with open(src_token_path, 'rb') as f:
                src_ids = pickle.load(f)

            with open(tgt_token_path, 'rb') as f:
                tgt_ids = pickle.load(f)

            return src_ids, tgt_ids

        with open(self.src_file, 'r', encoding='utf-8') as file:
            src_sentences = file.read().split('\n')

        with open(self.tgt_file, 'r', encoding='utf-8') as file:
            tgt_sentences = file.read().split('\n')

        # Check if directory exists and create if not
        directory_path = "tf-record"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        src_tokens = [self.tokenizer.tokenize(sentence) for sentence in src_sentences]
        tgt_tokens = [self.tokenizer.tokenize(sentence) for sentence in tgt_sentences]

        src_ids = [self.tokenizer.encode(tokens, truncation=True, padding='max_length', max_length=128)
                   for tokens in src_tokens if tokens]

        unique_tgt_tokens = set(token for tokens in tgt_tokens for token in tokens)
        tgt_vocab = {token: idx for idx, token in enumerate(unique_tgt_tokens)}

        # Ensure BERT's special tokens are present in the custom vocab
        for special_token in ['[PAD]', '[UNK]', '[CLS]', '[SEP]']:
            if special_token not in tgt_vocab:
                tgt_vocab[special_token] = len(tgt_vocab)

        # Map target tokens to their respective IDs
        tgt_ids = [[tgt_vocab[token] if token in tgt_vocab else tgt_vocab['[UNK]']
                    for token in tokens][:128] + [tgt_vocab['[PAD]']] * (128 - len(tokens))
                   for tokens in tgt_tokens]

        flat_src_ids = [item for sublist in src_ids for item in sublist]
        flat_tgt_ids = [item for sublist in tgt_ids for item in sublist]
        src_max_id = max(flat_src_ids)
        print(f"Source Maximum token ID: {src_max_id}")
        tgt_max_id = max(flat_tgt_ids)
        print(f"Target Maximum token ID: {tgt_max_id}")
        # if max_id >= len(self.tokenizer.vocab):
        #     print("Warning: There are token IDs that exceed the tokenizer's vocabulary size.")

        self._generate_vocab_file(src_tokens, flat_src_ids, self.src_file)
        self._generate_vocab_file(tgt_tokens, flat_tgt_ids, self.tgt_file)

        # Save tokenized data after tokenizing
        self.save_tokenized_data(src_ids, tgt_ids, src_token_path, tgt_token_path)

        return src_ids, tgt_ids

    def _generate_vocab_file(self, tokens, ids, filepath):
        """
        Generates a vocab file from the tokens


        :param tokens: list of tokenized sentences
        :param ids: list of ids.
        :param filepath: path to determine the output filename based on its extension
        """
        global vocab_file

        # Flatten the tokens list to get individual tokens
        flat_tokens = set(word for sentence in tokens for word in sentence)

        # If the file is English, use the Huggingface's vocabulary
        if filepath.endswith('.en'):
            if self.src_vocab_file.endswith('.en'):
                vocab_file = self.src_vocab_file
            elif self.tgt_vocab_file.endswith('.en'):
                vocab_file = self.tgt_vocab_file

            max_id = max(ids)

            # # Save the tokenizer's vocabulary instead of generating from tokens
            # with open(vocab_file, 'w', encoding='utf-8') as file:
            #     for word, id in self.tokenizer.vocab.items():
            #         # Only write tokens that are also in the tokens parameter
            #         if word in flat_tokens:
            #             file.write(f"{word} {id}\n")
            # Save the tokenizer's vocabulary instead of generating from tokens
            with open(vocab_file, 'w', encoding='utf-8') as file:
                for word, id in self.tokenizer.vocab.items():
                    # Only write tokens that are also in the tokens parameter
                    if id <= max_id:
                        file.write(f"{word} {id}\n")
        elif filepath.endswith('.am'):
            unique_tokens = flat_tokens
            token_to_id = {token: i for i, token in enumerate(unique_tokens)}

            if self.src_vocab_file.endswith('.am'):
                vocab_file = self.src_vocab_file
            elif self.tgt_vocab_file.endswith('.am'):
                vocab_file = self.tgt_vocab_file

            if os.path.exists(vocab_file):
                print(f"Vocab file {vocab_file} already exists. Skipping saving...")
                return

            # Generate and save vocabulary from tokens for Amharic
            with open(vocab_file, 'w', encoding='utf-8') as file:
                for word, id in token_to_id.items():
                    file.write(f"{word} {id}\n")
        else:
            return  # or raise an exception if neither .en or .am

    def save_tokenized_data(self, src_ids, tgt_ids, src_token_path, tgt_token_path):
        """
        Save the tokenized source and target data to disk.

        :param src_ids: Tokenized source sentences
        :param tgt_ids: Tokenized target sentences
        :param src_token_path: source token path
        :param tgt_token_path: target token path

        Args:
            src_token_path:
        """
        with open(src_token_path, 'wb') as f:
            pickle.dump(src_ids, f)

        with open(tgt_token_path, 'wb') as f:
            pickle.dump(tgt_ids, f)

    def load_tokenized_data(self, src_token_path, tgt_token_path):
        """
        Load the tokenized source and target data from disk.

        :param src_token_path: source token path
        :param tgt_token_path: target token path
        :return: tuple of lists, tokenized source and target sentences.
        """
        with open(src_token_path, 'rb') as f:
            src_ids = pickle.load(f)

        with open(tgt_token_path, 'rb') as f:
            tgt_ids = pickle.load(f)

        return src_ids, tgt_ids

    def pad(self):
        global src_pad, tgt_pad

        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized. Ensure that 'tokenize' method is called before 'pad'.")

        # For the English tokenizer
        if self.src_file.endswith('.en'):
            src_pad = self.tokenizer.pad_token_id if hasattr(self.tokenizer,
                                                             'pad_token_id') else self.tokenizer.eos_id()

        # For the Amharic tokenizer
        if self.tgt_file.endswith('.am'):
            # Change this line if you have a different tokenizer for Amharic
            tgt_pad = self.tokenizer.pad_token_id if hasattr(self.tokenizer,
                                                             'pad_token_id') else self.tokenizer.eos_id()
        return src_pad, tgt_pad

    def eos(self):
        global src_eos, tgt_eos
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized. Ensure that 'tokenize' method is called before 'pad'.")

        # For the English tokenizer
        if self.src_file.endswith('.en'):
            if hasattr(self.tokenizer, 'eos_token_id'):
                src_eos = self.tokenizer.eos_token_id
            else:
                raise ValueError("The English tokenizer does not have an 'eos_token_id' attribute.")

        # For the Amharic tokenizer
        if self.tgt_file.endswith('.am'):
            if hasattr(self.tokenizer, 'eos_token_id'):
                tgt_eos = self.tokenizer.eos_token_id
            else:
                raise ValueError("The Amharic tokenizer does not have an 'eos_token_id' attribute.")

        return src_eos, tgt_eos

    def unk(self):
        global src_unk, tgt_unk

        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized. Ensure that 'tokenize' method is called before 'pad'.")

        # For the English tokenizer
        if self.src_file.endswith('.en'):
            if hasattr(self.tokenizer, 'unk_token_id'):
                src_unk = self.tokenizer.unk_token_id
            else:
                raise ValueError("The English tokenizer does not have an 'unk_token_id' attribute.")

        # For the Amharic tokenizer
        if self.tgt_file.endswith('.am'):
            if hasattr(self.tokenizer, 'unk_token_id'):
                tgt_unk = self.tokenizer.unk_token_id
            else:
                raise ValueError("The Amharic tokenizer does not have an 'unk_token_id' attribute.")

        return src_unk, tgt_unk

    def _load_vocab_from_file(self, filepath):
        """
        Load the vocabulary from the provided file and return its size.

        :param filepath: Path to the vocab file.
        :return: int, size of the vocabulary.
        """
        with open(filepath, 'r', encoding='utf-8') as file:
            return len(file.readlines())

    def get_vocab_size(self):
        """
        Loads the vocabularies for both English and Amharic.

        :return: tuple of int, sizes of the English and Amharic vocabularies.
        """
        global src_vocab_size, tgt_vocab_size
        if not os.path.exists(self.src_vocab_file) or not os.path.exists(self.tgt_vocab_file):
            _, _ = self.tokenize()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        if self.src_vocab_file.endswith('.en'):
            src_vocab_size = self._load_vocab_from_file(self.src_vocab_file)
        elif self.src_vocab_file.endswith('.am'):
            src_vocab_size = self._load_vocab_from_file(self.src_vocab_file)
        if self.tgt_vocab_file.endswith('.en'):
            tgt_vocab_size = self._load_vocab_from_file(self.tgt_vocab_file)
        elif self.tgt_vocab_file.endswith('.am'):
            tgt_vocab_size = self._load_vocab_from_file(self.tgt_vocab_file)

        return src_vocab_size + 1, tgt_vocab_size + 1

    def get_bucketed_batches(self, task):
        """
        Gets batches of data with bucketing.

        :param task: task to be performed.
        :return: A dataset of bucketed batches.
        """
        src_token_path = f'src_{task}_tokens.pkl'
        tgt_token_path = f'tgt_{task}_tokens.pkl'
        # Try loading tokenized data
        try:
            src_ids, tgt_ids = self.load_tokenized_data(src_token_path, tgt_token_path)
        except FileNotFoundError:
            src_ids, tgt_ids = self.tokenize(task)

        src_ids = tf.ragged.constant(src_ids)
        tgt_ids = tf.ragged.constant(tgt_ids)

        src_ids = src_ids.to_tensor()
        tgt_ids = tgt_ids.to_tensor()

        max_length = max(max(len(seq) for seq in src_ids), max(len(seq) for seq in tgt_ids))

        # Define the bucket boundaries and the batch sizes for each bucket
        boundaries = [10, 20, 50, 80, 110, 140, 170, 200, 230, 260, 290, 320, 350, 380, 410, max_length + 10]
        batch_sizes = [self.batch_size] * (len(boundaries) + 1)

        def element_length_fn(src, tgt):
            return tf.shape(src)[0]

        print("source ids", len(src_ids))
        print("target ids", len(tgt_ids))
        dataset = tf.data.Dataset.from_tensor_slices((src_ids, tgt_ids))

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





