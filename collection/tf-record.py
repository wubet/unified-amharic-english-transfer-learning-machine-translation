import os
import tensorflow as tf
import tensorflow_text as text
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
from absl import flags
from absl import app
import os
import tensorflow as tf
from bert import bert_tokenization
import argparse

FLAGS = flags.FLAGS

flags.DEFINE_list(
    'source_dir', None, 'Directory of files storing source language '
                        'sequences.')
flags.DEFINE_list(
    'target_dir', None, 'directory of files storing target language '
                        'sequences.')
flags.DEFINE_string(
    'output_dir', None, 'Directory of final tf record.')
flags.DEFINE_boolean(
    'use_exist_vocab', None, 'using existing vocab or not')
flags.DEFINE_boolean(
    'do_lower_case', None, 'do use lower case or not')
flags.DEFINE_integer(
    'max_vocab_size', 32768, 'The desired vocabulary size. Ignored if '
                             '`min_count` is not None.')


# def create_vocab(source_dir, target_dir, output_dir, target_vocab_size, use_exist_vocab=False):
#     # Check if output directory exists, if not create it
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     vocab_file_path = os.path.join(output_dir, 'vocab.txt')
#
#     if use_exist_vocab and os.path.isfile(vocab_file_path):
#         with open(vocab_file_path, 'r') as f:
#             vocab = [line.strip() for line in f]
#         print(f'Vocabulary loaded from {vocab_file_path}')
#     else:
#         # Create a list to hold all file paths
#         file_paths = []
#
#         # source_files = [tf.io.gfile.GFile(fn) for fn in source_dir]
#         # target_files = [tf.io.gfile.GFile(fn) for fn in target_dir]
#
#         # Add input files
#         for file_path in source_dir:
#             directory = os.path.dirname(file_path)
#             filename = os.path.basename(file_path)
#             if filename.endswith('.txt') or filename.endswith('.en') or filename.endswith('.am'):
#                 file_paths.append(os.path.join(directory, filename))
#
#         # # Add target files
#         for file_path in target_dir:
#             directory = os.path.dirname(file_path)
#             filename = os.path.basename(file_path)
#             if filename.endswith('.txt') or filename.endswith('.en') or filename.endswith('.am'):
#                 file_paths.append(os.path.join(directory, filename))
#
#         # # Add input files
#         # for file_path in source_dir:
#         #     file_paths.append(file_path)
#         #
#         # for file_path in target_dir:
#         #     file_paths.append(file_path)
#
#         # Read the data from the files
#         # dataset = (tf.data.TextLineDataset(file_paths)  # Load text lines from files
#         #            .map(lambda line: tf.strings.split([line]).values)  # Split lines into words
#         #            )
#
#         # # Read the data from the files
#         # dataset = (tf.data.TextLineDataset(map(read_file_ignore_errors, file_paths))  # Load text lines from files
#         #            .map(lambda line: tf.strings.split([line]).values)  # Split lines into words
#         #            )
#
#         # Using map() to apply read_file_ignore_errors function to each filepath
#         # And then converting the map object to a list
#         file_paths_sanitized = list(map(read_file_ignore_errors, file_paths))
#
#         # Creating a dataset
#         dataset = tf.data.TextLineDataset(file_paths_sanitized)
#
#         # Split lines into words
#         dataset = dataset.map(lambda line: tf.strings.split([line]).values)
#
#         # Use the BERT tokenizer to create the vocab
#         bert_tokenizer_params = dict(lower_case=True)
#         reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]
#
#         vocab = bert_vocab.bert_vocab_from_dataset(
#             dataset,
#             # tokenizer_params=bert_tokenizer_params,
#             vocab_size=target_vocab_size,
#             reserved_tokens=reserved_tokens)
#
#         # Write the vocab to a file
#         with open(vocab_file_path, 'w', encoding="utf8") as f:
#             for token in vocab:
#                 print(token, file=f)
#
#         print(f'Vocabulary created and saved at {vocab_file_path}')
#     return vocab
#
#
# def read_file_ignore_errors(file_path):
#     with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
#         return f.read()


def create_vocab(source_dir, target_dir, output_dir, use_exist_vocab=False, do_lower_case=True):
    max_vocab_size = FLAGS.max_vocab_size
    # Check if output directory exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # base_path = os.path.abspath(os.getcwd())
    # vocab_path = os.path.join(base_path, output_dir)
    #
    # vocab_file_path = os.path.join(vocab_path, 'vocab.txt')
    vocab_file_path = os.path.join(output_dir, 'vocab.txt')

    print("vocab_file_path", vocab_file_path)

    # Use os.path.basename to get the filename
    # vocab_file = os.path.basename(vocab_file_path)
    # print("vocab_path", vocab_path)

    # # Check if the file already exists
    if not os.path.exists(vocab_file_path):
        # Create the file
        open(vocab_file_path, 'w').close()

    if use_exist_vocab and os.path.isfile(vocab_file_path):
        with open(vocab_file_path, 'r') as f:
            vocab = [line.strip() for line in f]
        print(f'Vocabulary loaded from {vocab_file_path}')
    else:
        # Create a list to hold all file paths
        file_paths = []

    # Add input files
    for file_path in source_dir:
        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        if filename.endswith('.txt') or filename.endswith('.en') or filename.endswith('.am'):
            file_paths.append(os.path.join(directory, filename))

    # # Add target files
    for file_path in target_dir:
        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        if filename.endswith('.txt') or filename.endswith('.en') or filename.endswith('.am'):
            file_paths.append(os.path.join(directory, filename))

    # Create a list to hold all file paths
    # file_paths = [os.path.join(dir, f) for f in files]

    # # Using map() to apply read_file_ignore_errors function to each filepath
    # # And then converting the map object to a list
    # file_paths_sanitized = list(map(read_file_ignore_errors, file_paths))
    #
    # # Creating a dataset
    # dataset = tf.data.TextLineDataset(file_paths_sanitized)
    #
    # # Split lines into words
    # dataset = dataset.map(lambda line: tf.strings.split([line]).values)

    dataset = (tf.data.TextLineDataset(file_paths)  # Load text lines from files
               .map(lambda line: tf.strings.split([line]).values)  # Split lines into words
               )

    # Initialize the tokenizer
    tokenizer = bert_tokenization.FullTokenizer(vocab_file_path, do_lower_case)

    # Create the vocab
    vocab = tokenizer.vocab  # This is a dictionary

    # If the vocab is smaller than the maximum size, update it
    if len(vocab) < max_vocab_size:
        for batch in dataset:
            for token in batch:
                if token.numpy() not in vocab:
                    vocab[token.numpy()] = len(vocab)  # Add the new token to the vocab

        # If the vocab is too large, truncate it
        if len(vocab) > max_vocab_size:
            vocab = dict(list(vocab.items())[:max_vocab_size])

        # Write the vocab to a file
        with open(vocab_file_path, 'w') as f:
            for token, index in vocab.items():
                f.write(f'{token}\t{index}\n')

        print(f'Vocabulary created and saved at {vocab_file_path}')

    return vocab


def read_file_ignore_errors(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


# def main(argv):
#     if FLAGS.source_dir is None:
#         raise ValueError('The source directory is required')
#     if FLAGS.target_dir is None:
#         raise ValueError('The target directory is required')
#     if FLAGS.output_dir is None:
#         raise ValueError('The output directory is required')
#
#     source_dir = FLAGS.source_dir[0] if isinstance(FLAGS.source_dir, list) else FLAGS.source_dir
#     target_dir = FLAGS.target_dir[0] if isinstance(FLAGS.target_dir, list) else FLAGS.target_dir
#     output_dir = FLAGS.output_dir if isinstance(FLAGS.output_dir, str) else str(FLAGS.output_dir)
#     output_dir = FLAGS.output_dir if isinstance(FLAGS.output_dir, str) else str(FLAGS.output_dir)
#     use_exist_vocab = True if FLAGS.use_exist_vocab == 1 else False
#     target_vocab_size = FLAGS.target_vocab_size if FLAGS.target_vocab_size is not None else 32768
#
#     create_vocab(source_dir, target_dir, output_dir, use_exist_vocab, target_vocab_size)
#
#
# if __name__ == "__main__":
#     app.run(main)
def main(argv):
    del argv  # Unused.

    if not FLAGS.source_dir:
        raise ValueError('You must specify "source_dir"!')
    if not FLAGS.target_dir:
        raise ValueError('You must specify "target_dir"!')
    if not FLAGS.output_dir:
        raise ValueError('You must specify "output_dir"!')

    create_vocab(FLAGS.source_dir, FLAGS.target_dir, FLAGS.output_dir, FLAGS.use_exist_vocab, FLAGS.do_lower_case)


if __name__ == '__main__':
    flags.mark_flag_as_required('source_dir')
    flags.mark_flag_as_required('target_dir')
    flags.mark_flag_as_required('output_dir')
    app.run(main)
