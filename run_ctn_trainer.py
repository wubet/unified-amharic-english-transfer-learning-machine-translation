from absl import flags
from absl import app
from transformers import BertConfig, TFBertModel
from ctn_tranformer_model.ctn_model_runner import TrainCtnmtModel
from data.language_processor import *
from models.model import *

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'src_lang_file_path', None, 'Directory of files storing Source language '
                                'sequences.')
flags.DEFINE_string(
    'tgt_lang_file_path', None, 'directory of files storing Target language '
                                'sequences.')
flags.DEFINE_string(
    'src_vocab_file_path', None, 'directory of final storing Source language vocab.')
flags.DEFINE_string(
    'tgt_vocab_file_path', None, 'directory of final storing Target language vocab.')
flags.DEFINE_string(
    'pre_trained_model_path', None, 'pre_trained model path.')
flags.DEFINE_string(
    'check_point_path', None, 'check point path.')
flags.DEFINE_string(
    'source_language', None, 'source language.')
flags.DEFINE_string(
    'target_language', None, 'target language.')
flags.DEFINE_integer(
    'num_layers', 6, 'Num of layers in encoder stack.')
flags.DEFINE_integer(
    'd_model', 768, 'The dimensionality of the embedding vector.')
flags.DEFINE_integer(
    'num_heads', 8, 'Num of attention heads.')
flags.DEFINE_integer(
    'dff', 2048, 'The depth of the intermediate dense layer of the'
                 'feed-forward sublayer.')
flags.DEFINE_float(
    'learning_rate', 0.001, 'Base learning rate.')

flags.DEFINE_float(
    'dropout_rate', 0.1, 'Dropout rate for the Dropout layers.')

flags.DEFINE_integer(
    'max_num_tokens', 4096, 'The maximum num of tokens in each batch.')
flags.DEFINE_integer(
    'max_length', 64, 'Source or target seqs longer than this will be filtered'
                      ' out.')
flags.DEFINE_integer(
    'num_parallel_calls', 8, 'Num of TFRecord files to be processed '
                             'concurrently.')
flags.DEFINE_integer(
    'batch_size', 10, 'batch size')
flags.DEFINE_integer(
    'num_steps', 500000, 'Num of training iterations (minibatches).')
flags.DEFINE_integer(
    'save_ckpt_per_steps', 5000, 'Every this num of steps to save checkpoint.')
flags.DEFINE_integer(
    'log_per_iterations', 100, 'Every this num of steps to save checkpoint.')


def main(argv):
    if not FLAGS.src_lang_file_path:
        raise ValueError('You must specify "Source language directory"!')
    if not FLAGS.tgt_lang_file_path:
        raise ValueError('You must specify "Target language directory"!')
    if not FLAGS.src_vocab_file_path:
        raise ValueError('You must specify "Source language vocab directory"!')
    if not FLAGS.tgt_vocab_file_path:
        raise ValueError('You must specify "Target language vocab directory"!')
    if not FLAGS.pre_trained_model_path:
        raise ValueError('You must specify "pre_trained model path"!')
    if not FLAGS.check_point_path:
        raise ValueError('You must specify "check_point path"!')
    if not FLAGS.source_language:
        raise ValueError('You must specify "source language"!')
    if not FLAGS.target_language:
        raise ValueError('You must specify "target language"!')

    src_lang_file_path = FLAGS.src_lang_file_path
    tgt_lang_file_path = FLAGS.tgt_lang_file_path
    src_vocab_file_path = FLAGS.src_vocab_file_path
    tgt_vocab_file_path = FLAGS.tgt_vocab_file_path
    pre_trained_model_path = FLAGS.pre_trained_model_path
    check_point_path = FLAGS.check_point_path
    source_language = FLAGS.source_language
    target_language = FLAGS.target_language
    d_model = FLAGS.d_model
    num_layers = FLAGS.num_layers
    num_heads = FLAGS.num_heads
    dff = FLAGS.dff
    dropout_rate = FLAGS.dropout_rate
    learning_rate = FLAGS.learning_rate
    # epochs = FLAGS.epochs # Set your desired number of epochs
    batch_size = FLAGS.batch_size
    num_steps = FLAGS.num_steps
    save_ckpt_per_steps = FLAGS.save_ckpt_per_steps
    log_per_iterations = FLAGS.log_per_iterations

    language_processor = LanguageProcessor(src_lang_file_path, tgt_lang_file_path, src_vocab_file_path,
                                           tgt_vocab_file_path, batch_size)

    _, _ = language_processor.tokenize("training")

    # Load vocab
    src_vocab_size, tgt_vocab_size = language_processor.get_vocab_size()

    # Tokenize and get batches of data
    train_dataset = language_processor.get_bucketed_batches("training")

    # # Load the teacher model
    teacher_model_path = pre_trained_model_path  # Path to the pretrained model

    transformer = Transformer(
        num_layers, d_model, num_heads, dff,
        src_vocab_size, tgt_vocab_size,
        src_vocab_size, tgt_vocab_size, dropout_rate)

    # If a teacher model path is provided, load the teacher model (BERT).
    teacher_model = None
    if teacher_model_path:
        # Create a BERT configuration with output_hidden_states set to True
        config = BertConfig.from_pretrained(teacher_model_path)
        config.output_hidden_states = True
        teacher_model = TFBertModel.from_pretrained(teacher_model_path, config=config)

    train_ctnmt_model = TrainCtnmtModel(transformer,
                                        teacher_model,
                                        check_point_path,
                                        learning_rate,
                                        d_model,
                                        num_steps,
                                        save_ckpt_per_steps,
                                        log_per_iterations,
                                        source_language,
                                        target_language)

    train_ctnmt_model.train(train_dataset)


if __name__ == '__main__':
    flags.mark_flag_as_required('src_lang_file_path')
    flags.mark_flag_as_required('tgt_lang_file_path')
    flags.mark_flag_as_required('src_vocab_file_path')
    flags.mark_flag_as_required('tgt_vocab_file_path')
    flags.mark_flag_as_required('pre_trained_model_path')
    flags.mark_flag_as_required('check_point_path')
    flags.mark_flag_as_required('source_language')
    flags.mark_flag_as_required('target_language')
    app.run(main)
