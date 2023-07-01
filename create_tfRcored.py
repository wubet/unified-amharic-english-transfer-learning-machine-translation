import os
import tensorflow as tf
import tensorflow_text as text
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
import argparse


def create_vocab(source_dir, target_dir, output_dir, target_vocab_size, use_exist_vocab):
    # Check if output directory exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    vocab_file_path = os.path.join(output_dir, 'vocab.txt')

    if use_exist_vocab and os.path.isfile(vocab_file_path):
        with open(vocab_file_path, 'r') as f:
            vocab = [line.strip() for line in f]
        print(f'Vocabulary loaded from {vocab_file_path}')
    else:
        # Create a list to hold all file paths
        file_paths = []

        # Add input files
        for filename in os.listdir(source_dir):
            if filename.endswith('.txt'):
                file_paths.append(os.path.join(source_dir, filename))

        # Add target files
        for filename in os.listdir(target_dir):
            if filename.endswith('.txt'):
                file_paths.append(os.path.join(target_dir, filename))

        # Read the data from the files
        dataset = (tf.data.TextLineDataset(file_paths)  # Load text lines from files
                   .map(lambda line: tf.strings.split([line]).values)  # Split lines into words
                   )

        # Use the BERT tokenizer to create the vocab
        bert_tokenizer_params = dict(lower_case=True)
        reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]

        vocab = bert_vocab.bert_vocab_from_dataset(
            dataset,
            tokenizer_params=bert_tokenizer_params,
            vocab_size=target_vocab_size,
            reserved_tokens=reserved_tokens)

        # Write the vocab to a file
        with open(vocab_file_path, 'w') as f:
            for token in vocab:
                print(token, file=f)

        print(f'Vocabulary created and saved at {vocab_file_path}')
    return vocab


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Create Vocabulary from text files")

    # Add the arguments
    parser.add_argument("source_dir", type=str, help="The directory containing the source text files")
    parser.add_argument("target_dir", type=str, help="The directory containing the target text files")
    parser.add_argument("output_dir", type=str, help="The directory to output the vocabulary file")
    parser.add_argument("target_vocab_size", type=int, help="The desired vocabulary size")
    parser.add_argument("--use_exist_vocab", action="store_true", help="Use existing vocabulary if it exists")

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the arguments
    create_vocab(args.source_dir, args.target_dir, args.output_dir, args.target_vocab_size, args.use_exist_vocab)


if __name__ == "__main__":
    main()
