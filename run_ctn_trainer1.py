from absl import flags
from absl import app
from ctn_tranformer_model.ctn_transformer import CtnTransformer
from ctn_tranformer_model.ctn_model_runner import TrainCntModel
from data.language_processor import *
from data.sentence_toknizer import *

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'eng_lang_file_path', None, 'Directory of files storing English language '
                                'sequences.')
flags.DEFINE_string(
    'amh_lang_file_path', None, 'directory of files storing Amharic language '
                                'sequences.')
flags.DEFINE_string(
    'eng_vocab_file_path', None, 'directory of final storing English language vocab.')

flags.DEFINE_string(
    'amh_vocab_file_path', None, 'directory of final storing Amharic language vocab.')

flags.DEFINE_string(
    'pre_trained_model_path', None, 'pre_trained model path.')


def main(argv):
    del argv  # Unused.

    if not FLAGS.eng_lang_file_path:
        raise ValueError('You must specify "English language directory"!')
    if not FLAGS.amh_lang_file_path:
        raise ValueError('You must specify "Amharic language directory"!')
    if not FLAGS.eng_vocab_file_path:
        raise ValueError('You must specify "English language vocab directory"!')
    if not FLAGS.amh_vocab_file_path:
        raise ValueError('You must specify "Amharic language vocab directory"!')
    if not FLAGS.pre_trained_model_path:
        raise ValueError('You must specify "pre_trained model path"!')

    eng_lang_file_path = FLAGS.eng_lang_file_path
    amh_lang_file_path = FLAGS.amh_lang_file_path
    eng_vocab_file_path = FLAGS.eng_vocab_file_path
    amh_vocab_file_path = FLAGS.amh_vocab_file_path
    pre_trained_model_path = FLAGS.pre_trained_model_path
    buffer_size = 1000
    batch_size = 10

    processor = LanguageProcessor(eng_lang_file_path, amh_lang_file_path, eng_vocab_file_path,
                                  amh_vocab_file_path, batch_size)

    # Tokenize and get batches of data
    dataset = processor.get_bucketed_batches()

    # Load vocab
    eng_vocab_size, amh_vocab_size = processor.load_vocab()

    # Define the hypermarkets
    d_model = 768
    num_layers = 6
    num_heads = 8
    dff = 2048
    dropout_rate = 0.1
    learning_rate = 2.0

    # # Load the teacher model
    teacher_model_path = pre_trained_model_path  # Path to the pretrained model

    # Initialize the transformer model for transfer learning
    cnt_transformer = CtnTransformer(
        # input_vocab_size=source_tokenizer.vocab_size + 2,
        # target_vocab_size=target_tokenizer.vocab_size + 2,
        input_vocab_size=eng_vocab_size,
        target_vocab_size=amh_vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dff=dff,
        dropout_rate=dropout_rate,
        teacher_model_path=teacher_model_path)

    # Create the Transformer model
    cnt_transformer.create_transformer()

    # Create the metrics for monitoring training
    cnt_transformer.create_metrics()

    # Initialize the training class with the transformer model
    train_cnt_model = TrainCntModel(cnt_transformer)

    cnt_transformer.transformer_model.summary()

    # Assuming `dataset` is your tf.data.Dataset object with inputs and targets
    epochs = 2  # Set your desired number of epochs
    train_cnt_model.train(dataset, epochs)


if __name__ == '__main__':
    flags.mark_flag_as_required('eng_lang_file_path')
    flags.mark_flag_as_required('amh_lang_file_path')
    flags.mark_flag_as_required('eng_vocab_file_path')
    flags.mark_flag_as_required('amh_vocab_file_path')
    flags.mark_flag_as_required('pre_trained_model_path')
    app.run(main)
