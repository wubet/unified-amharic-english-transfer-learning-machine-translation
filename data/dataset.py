# import tensorflow as tf
# import tensorflow_datasets as tfds
# import os
#
#
# def download_dataset(tfds_address, disable_progress_bar=False):
#     if disable_progress_bar:
#         tfds.disable_progress_bar()
#     data, metadata = tfds.load(tfds_address, with_info=True, as_supervised=True)
#     train_data, val_data = data['train'], data['validation']
#     return train_data, val_data
#
#
# def get_dataset(src_file_path, tgt_file_path):
#     # Load the source and target files
#     src_data = tf.data.TextLineDataset(src_file_path)
#     tgt_data = tf.data.TextLineDataset(tgt_file_path)
#
#     # Pair the source and target sentences
#     dataset = tf.data.Dataset.zip((src_data, tgt_data))
#
#     return dataset
#
#
# def create_tokenizers(dataset, approx_vocab_size=2 ** 13):
#     source_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
#         (src.numpy() for src, tgt in dataset),
#         target_vocab_size=approx_vocab_size
#     )
#     target_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
#         (tgt.numpy() for src, tgt in dataset),
#         target_vocab_size=approx_vocab_size
#     )
#     return source_tokenizer, target_tokenizer
#
#
# class DataLoader:
#
#     def __init__(self, source_tokenizer, target_tokenizer, max_limit=40):
#         self.source_tokenizer = source_tokenizer
#         self.target_tokenizer = target_tokenizer
#         self.max_limit = max_limit
#
#     def preprocess(self, language_1, language_2):
#         language_1 = [
#                          self.source_tokenizer.vocab_size
#                      ] + self.source_tokenizer.encode(
#             language_1.numpy()
#         ) + [
#                          self.source_tokenizer.vocab_size + 1
#                      ]
#         language_2 = [
#                          self.target_tokenizer.vocab_size
#                      ] + self.target_tokenizer.encode(
#             language_2.numpy()
#         ) + [
#                          self.target_tokenizer.vocab_size + 1
#                      ]
#         return language_1, language_2
#
#     def map_function(self, language_1, language_2):
#         language_1, language_2 = tf.py_function(
#             self.preprocess,
#             [language_1, language_2],
#             [tf.int64, tf.int64]
#         )
#         language_1.set_shape([None])
#         language_2.set_shape([None])
#         return language_1, language_2
#
#     def filter_max_length(self, x, y):
#         return tf.logical_and(
#             tf.size(x) <= self.max_limit,
#             tf.size(y) <= self.max_limit
#         )
#
#     def get_dataset(self, dataset, buffer_size, batch_size):
#         tf_dataset = dataset.map(self.map_function)
#         if self.max_limit is not None:
#             tf_dataset = tf_dataset.filter(self.filter_max_length)
#         tf_dataset = tf_dataset.cache()
#         tf_dataset = tf_dataset.shuffle(buffer_size)
#         tf_dataset = tf_dataset.padded_batch(
#             batch_size, padded_shapes=([None], [None])
#         )
#         tf_dataset = tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)
#         return tf_dataset
#
#     def save_vocabs_to_file(self, source_tokenizer, vocab_filename):
#         source_tokenizer.save_to_file(vocab_filename)
#
#     def get_tokenizers(self, src_vocab_file_path, tgt_vocab_file_path, dataset):
#
#         # Load the tokenizers from the provided file paths
#         try:
#             self.source_tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(src_vocab_file_path)
#             self.target_tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(tgt_vocab_file_path)
#         # If the files do not exist, create the tokenizers
#         except:
#             self.source_tokenizer, self.target_tokenizer = create_tokenizers(dataset)
#             self.save_vocabs_to_file(self.source_tokenizer, src_vocab_file_path)
#             self.save_vocabs_to_file(self.target_tokenizer, tgt_vocab_file_path)
#         return self.source_tokenizer, self.target_tokenizer


