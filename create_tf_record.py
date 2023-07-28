from absl import flags
from absl import app
import os
import tensorflow as tf
from bert import bert_tokenization
from typing import List

FLAGS = flags.FLAGS

flags.DEFINE_list(
    'language_dir', None, 'Directory of files storing source language '
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


# create vocabulary files from source and target language file
def create_vocab(language_dir: List[str], output_file, use_exist_vocab=False, do_lower_case=False, max_vocab_size=32768):
    # Check if output directory exists, if not create it
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check if the file already exists
    if not os.path.exists(output_file):
        # Create the file
        open(output_file, 'w').close()

    if use_exist_vocab and os.path.isfile(output_file):
        with open(output_file, 'r') as f:
            vocab = [line.strip().split('\t')[0] for line in f]
        print(f'Vocabulary loaded from {output_file}')
    else:
        # Create a list to hold all file paths
        file_paths = []
        # Add files
        for file_path in language_dir:
            directory = os.path.dirname(file_path)
            filename = os.path.basename(file_path)
            if filename.endswith('.txt') or filename.endswith('.en') or filename.endswith('.am'):
                file_paths.append(os.path.join(directory, filename))

        dataset = (tf.data.TextLineDataset(file_paths)  # Load text lines from files
                   .map(lambda line: tf.strings.split([line]).values)  # Split lines into words
                   )

        # Initialize the tokenizer
        tokenizer = bert_tokenization.FullTokenizer(output_file, do_lower_case)

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
            with open(output_file, 'w') as f:
                for token, index in vocab.items():
                    f.write(f'{token}\t{index}\n')

            print(f'Vocabulary created and saved at {output_file}')

    return vocab


def main(argv):
    del argv  # Unused.

    if not FLAGS.language_dir:
        raise ValueError('You must specify "source or target language directory"!')
    if not FLAGS.output_dir:
        raise ValueError('You must specify "output directory"!')

    create_vocab(FLAGS.language_dir, FLAGS.output_dir, FLAGS.use_exist_vocab, FLAGS.do_lower_case, FLAGS.max_vocab_size)


if __name__ == '__main__':
    flags.mark_flag_as_required('language_dir')
    flags.mark_flag_as_required('output_dir')
    app.run(main)
