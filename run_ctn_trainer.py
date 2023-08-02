from absl import flags
from absl import app
from ctn_tranformer_model.ctn_transformer import CtnTransformer
from ctn_tranformer_model.ctn_model_runner import TrainCntModel
# from data.dataset import *
from data.sentence_toknizer import *

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'src_lang_file_path', None, 'Directory of files storing source language '
                                'sequences.')
flags.DEFINE_string(
    'tgt_lang_file_path', None, 'directory of files storing target language '
                                'sequences.')
flags.DEFINE_string(
    'src_vocab_file_path', None, 'directory of final storing source language vocab.')

flags.DEFINE_string(
    'tgt_vocab_file_path', None, 'directory of final storing target language vocab.')

flags.DEFINE_string(
    'pre_trained_model_path', None, 'pre_trained model path.')


def main(argv):
    del argv  # Unused.

    if not FLAGS.src_lang_file_path:
        raise ValueError('You must specify "source language directory"!')
    if not FLAGS.tgt_lang_file_path:
        raise ValueError('You must specify "target language directory"!')
    if not FLAGS.src_vocab_file_path:
        raise ValueError('You must specify "source language vocab directory"!')
    if not FLAGS.tgt_vocab_file_path:
        raise ValueError('You must specify "target language vocab directory"!')
    if not FLAGS.pre_trained_model_path:
        raise ValueError('You must specify "pre_trained model path"!')

    src_lang_file_path = FLAGS.src_lang_file_path
    tgt_lang_file_path = FLAGS.tgt_lang_file_path
    src_vocab_file_path = FLAGS.src_vocab_file_path
    tgt_vocab_file_path = FLAGS.tgt_vocab_file_path
    pre_trained_model_path = FLAGS.pre_trained_model_path
    buffer_size = 1000
    batch_size = 100

    # Assuming we are using a preprocessed TensorFlow dataset
    # train_dataset_path = "path_to_your_train_dataset"
    # train_dataset = tf.data.experimental.load(train_dataset_path)

    # data_loader = DataLoader(source_tokenizer, target_tokenizer, max_limit=40)
    # data_loader.save_vocabs_to_file("source_vocab.txt", "target_vocab.txt")
    # dataset = get_dataset("source_sentences.txt", "target_sentences.txt")

    # dataset = get_dataset(src_lang_file_path, tgt_lang_file_path)
    # data_loader = DataLoader(None, None)
    # source_tokenizer, target_tokenizer = data_loader.get_tokenizers(src_vocab_file_path, tgt_vocab_file_path, dataset)

    # data_loader.save_vocabs_to_file(src_vocab_file_path,   tgt_vocab_file_path)
    # tf_dataset = data_loader.get_dataset(dataset, buffer_size, batch_size)

    # src_vocab_file_path = source_tokenizer.vocab_size + 2
    # tgt_vocab_file_path = target_tokenizer.vocab_size + 2

    sentence_tokenizer = SentenceTokenizer(src_lang_file_path, tgt_lang_file_path, batch_size)
    #
    # # train_dataset = load_dataset(src_lang_file_path, tgt_lang_file_path)
    train_dataset = sentence_tokenizer.load_and_tokenize()

    # Define the hypermarkets
    d_model = 768
    num_layers = 6
    num_heads = 8
    dff = 2048
    dropout_rate = 0.1
    learning_rate = 2.0

    # # Load the teacher model
    teacher_model_path = pre_trained_model_path  # Path to the pretrained model

    # # Load the dataset to calculate the vocab sizes
    # tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    #     (data.numpy() for data, _ in train_dataset), target_vocab_size=2 ** 13)

    # # Compute the vocab sizes. +2 for <start> and <end> tokens
    # input_vocab_size = target_vocab_size = tokenizer.vocab_size + 2

    source_vocab = sentence_tokenizer.load_vocab(src_vocab_file_path)
    input_vocab_size = sentence_tokenizer.vocab_size(source_vocab)
    target_vocab = sentence_tokenizer.load_vocab(tgt_vocab_file_path)
    target_vocab_size = sentence_tokenizer.vocab_size(target_vocab)

    # Initialize the transformer model for transfer learning
    cnt_transformer = CtnTransformer(
        # input_vocab_size=source_tokenizer.vocab_size + 2,
        # target_vocab_size=target_tokenizer.vocab_size + 2,
        input_vocab_size=input_vocab_size,
        target_vocab_size= target_vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dff=dff,
        dropout_rate=dropout_rate,
        teacher_model_path=teacher_model_path)

    # Create the Transformer model
    cnt_transformer.create_transformer()

    # # Create the optimizer and loss object
    # cnt_transformer.create_optimizer(learning_rate=learning_rate)
    #
    # # Create loss function
    # cnt_transformer.create_cross_entropy()

    # Create the metrics for monitoring training
    cnt_transformer.create_metrics()

    # Initialize the training class with the transformer model
    train_cnt_model = TrainCntModel(cnt_transformer)

    cnt_transformer.transformer_model.summary()

    # Assuming `dataset` is your tf.data.Dataset object with inputs and targets
    epochs = 2  # Set your desired number of epochs
    train_cnt_model.train(train_dataset, epochs)


if __name__ == '__main__':
    flags.mark_flag_as_required('src_lang_file_path')
    flags.mark_flag_as_required('tgt_lang_file_path')
    flags.mark_flag_as_required('src_vocab_file_path')
    flags.mark_flag_as_required('tgt_vocab_file_path')
    flags.mark_flag_as_required('pre_trained_model_path')
    app.run(main)
