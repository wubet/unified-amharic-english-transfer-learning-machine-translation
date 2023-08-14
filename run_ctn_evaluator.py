from absl import flags
from absl import app
from transformers import BertConfig, TFBertModel
from ctn_tranformer_model.ctn_model_runner import TrainCtnmtModel
from data.language_processor import *
from evaluators.ctn_transformer_evaluator import CtnTransformerEvaluator
from models.model import *

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'num_layers', 6, 'Num of layers in stack.')
flags.DEFINE_integer(
    'd_model', 768, 'The dimensionality of the embedding vector.')
flags.DEFINE_integer(
    'num_heads', 8, 'Num of attention heads.')
flags.DEFINE_integer(
    'dff', 2048, 'The depth of the intermediate dense layer of the'
                 'feed-forward sublayer.')
flags.DEFINE_float(
    'dropout_rate', 0.1, 'Dropout rate for the Dropout layers.')
flags.DEFINE_integer(
    'extra_decode_length', 50, 'The max decode length would be'
                               ' the sum of `tgt_seq_len` and `extra_decode_length`.')
flags.DEFINE_integer(
    'beam_width', 4, 'Beam width for beam search.')
flags.DEFINE_float(
    'alpha', 0.6, 'The parameter for length normalization used in beam search.')
flags.DEFINE_integer(
    'decode_batch_size', 32, 'Number of sequences in a batch for inference.')
flags.DEFINE_integer(
    'src_max_length', 100, 'The number of tokens that source sequences will be '
                           'truncated or zero-padded to in inference mode.')

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
    'translation_output_filename', 'translations.txt', 'Path to the output '
                                                       'file that the translations will be written to.')
flags.DEFINE_string(
    'model_dir', None, 'Path to the directory that checkpoint files will be '
                       'written to.')
flags.DEFINE_string(
    'is_target_language_amharic', 'False', 'boolean value '
                                           'the target language is Amharic or not.')


def main(argv):
    del argv  # Unused.
    # Define the hypermarkets
    d_model = 768
    num_layers = 6
    num_heads = 8
    dff = 2048
    dropout_rate = 0.1
    learning_rate = 0.1
    epochs = 2  # Set your desired number of epochs
    batch_size = 10

    if not FLAGS.src_lang_file_path:
        raise ValueError('You must specify "Source language directory"!')
    if not FLAGS.src_vocab_file_path:
        raise ValueError('You must specify "Source language vocab directory"!')
    if not FLAGS.model_dir:
        raise ValueError('You must specify "Trained model directory"!')

    src_lang_file_path = FLAGS.src_lang_file_path
    tgt_lang_file_path = FLAGS.tgt_lang_file_path
    src_vocab_file_path = FLAGS.src_vocab_file_path
    tgt_vocab_file_path = FLAGS.tgt_vocab_file_path
    model_dir = FLAGS.model_dir

    num_layers = FLAGS.num_layers
    dff = FLAGS.dff
    num_heads = FLAGS.num_heads
    d_model = FLAGS.d_model
    dropout_rate = FLAGS.dropout_rate

    extra_decode_length = FLAGS.extra_decode_length
    beam_width = FLAGS.beam_width
    alpha = FLAGS.alpha
    decode_batch_size = FLAGS.decode_batch_size
    src_max_length = FLAGS.src_max_length

    source_text_filename = FLAGS.source_text_filename
    target_text_filename = FLAGS.target_text_filename
    translation_output_filename = FLAGS.translation_output_filename
    is_target_language_amharic = bool(FLAGS.is_target_language_amharic)

    processor = LanguageProcessor(src_lang_file_path, tgt_lang_file_path, src_vocab_file_path,
                                  tgt_vocab_file_path, batch_size)

    # Tokenize and get batches of data
    test_dataset = processor.get_bucketed_batches()

    # Load vocab
    src_vocab_size, tgt_vocab_size = processor.load_vocab()

    # # Load the teacher model
    # teacher_model_path = pre_trained_model_path  # Path to the pretrained model

    trained_model = Transformer(
        num_layers, d_model, num_heads, dff,
        src_vocab_size, tgt_vocab_size,
        src_vocab_size, tgt_vocab_size, dropout_rate)

    ckpt = tf.train.Checkpoint(model=trained_model)
    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    if latest_ckpt is None:
        raise ValueError('No checkpoint is found in %s' % model_dir)
    print('Loaded latest checkpoint ', latest_ckpt)
    ckpt.restore(latest_ckpt).expect_partial()

    # Create the evaluator instance
    evaluator = CtnTransformerEvaluator(trained_model, target_tokenizer)

    # Evaluate the model and get the BLEU score
    bleu_score = evaluator.evaluate(test_dataset)
    print("BLEU Score:", bleu_score)

    # # If a teacher model path is provided, load the teacher model (BERT).
    # teacher_model = None
    # if teacher_model_path:
    #     # Create a BERT configuration with output_hidden_states set to True
    #     config = BertConfig.from_pretrained(teacher_model_path)
    #     config.output_hidden_states = True
    #     teacher_model = TFBertModel.from_pretrained(teacher_model_path, config=config)
    #
    # train_ctnmt_model = TrainCtnmtModel(transformer, teacher_model, check_point_path,
    #                                     learning_rate, d_model, source_language, target_language)
    #
    # train_ctnmt_model.train(dataset, epochs)


if __name__ == '__main__':
    flags.mark_flag_as_required('src_lang_file_path')
    flags.mark_flag_as_required('src_vocab_file_path')
    flags.mark_flag_as_required('model_dir')
    app.run(main)
