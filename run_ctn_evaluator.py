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
flags.DEFINE_integer(
    'batch_size', 10, 'batch size')


def main(argv):
    # del argv  # Unused.
    # # Define the hypermarkets
    # d_model = 768
    # num_layers = 6
    # num_heads = 8
    # dff = 2048
    # dropout_rate = 0.1
    # learning_rate = 0.1
    # epochs = 2  # Set your desired number of epochs
    # batch_size = 10

    if not FLAGS.src_lang_file_path:
        raise ValueError('You must specify "Source language directory"!')
    if not FLAGS.tgt_lang_file_path:
        raise ValueError('You must specify "Target language directory"!')
    if not FLAGS.src_vocab_file_path:
        raise ValueError('You must specify "Source language vocab directory"!')
    if not FLAGS.tgt_vocab_file_path:
        raise ValueError('You must specify "Target language vocab directory"!')
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
    batch_size = FLAGS.batch_size

    # extra_decode_length = FLAGS.extra_decode_length
    # beam_width = FLAGS.beam_width
    # alpha = FLAGS.alpha
    # decode_batch_size = FLAGS.decode_batch_size
    # src_max_length = FLAGS.src_max_length

    language_processor = LanguageProcessor(src_lang_file_path, tgt_lang_file_path, src_vocab_file_path,
                                           tgt_vocab_file_path, batch_size)

    # Tokenize and get batches of data
    test_dataset = language_processor.get_bucketed_batches()

    # Load vocab
    src_vocab_size, tgt_vocab_size = language_processor.load_vocab()

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

    # Read and extract source and target language data from the file paths
    with open(src_lang_file_path, 'r', encoding='utf-8') as file:
        src_lang_data = file.read().split('\n')

    with open(tgt_lang_file_path, 'r', encoding='utf-8') as file:
        tgt_lang_data = file.read().split('\n')

    # Create the evaluator instance
    evaluator = CtnTransformerEvaluator(trained_model, language_processor, src_lang_data, tgt_lang_data)

    if tgt_lang_file_path is not None:
        # Evaluate the model and get the BLEU score
        bleu_score = evaluator.evaluate(test_dataset)
        print("BLEU Score:", bleu_score)
        # today = date.today()
        # data = {'Model_Name': ['Transformer'],
        #         'Case_sensitive_BLUE_score': [case_sensitive_score],
        #         'Case_insensitive_BLUE_score': [case_insensitive_score],
        #         'Date': [today]
        #         }
        # evaluation_file_path = os.path.join(current_dir, "tf-transformer/output/BLUE_evaluation.csv")
        #
        # df = pd.DataFrame(data)
        # if os.path.exists(evaluation_file_path):
        #     df.to_csv(evaluation_file_path, mode='a', header=False)
        # else:
        #     df.to_csv(evaluation_file_path, index=False, header=True)
        #
        # print('BLEU(case insensitive): %f' % case_insensitive_score)
        # print('BLEU(case sensitive): %f' % case_sensitive_score)

    # else:
        # evaluator.translate(
        #     source_text_filename, is_target_language_amharic, translation_output_filename)
        # print('Inference mode: no groundtruth translations.\nTranslations written '
        #       'to file "%s"' % translation_output_filename)


if __name__ == '__main__':
    flags.mark_flag_as_required('src_lang_file_path')
    flags.mark_flag_as_required('tgt_lang_file_path')
    flags.mark_flag_as_required('src_vocab_file_path')
    flags.mark_flag_as_required('tgt_vocab_file_path')
    flags.mark_flag_as_required('model_dir')
    app.run(main)
