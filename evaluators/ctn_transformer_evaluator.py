import tensorflow as tf
from keras.losses import SparseCategoricalCrossentropy
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

from common.noam_learning_rate_scheduler import NoamLearningRateSchedule
from common.utils import *


class CtnTransformerEvaluator:
    def __init__(self, model, language_processor, src_lang_data, tgt_lang_data, ckpts):
        self.model = model
        self.language_processor = language_processor
        self.src_lang_data = src_lang_data
        self.tgt_lang_data = tgt_lang_data
        self.checkpoint_dir = ckpts
        # self.tokenizer_src = language_processor.tokenizer_src
        # self.tokenizer_tgt = language_processor.tokenizer_tgt

    def evaluate(self, test_dataset):
        # Restore checkpoint
        learning_rate = NoamLearningRateSchedule(initial_factor=1.0, dmodel=768, warmup_steps=1000)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        checkpoint, _ = get_checkpoints(self.model, optimizer, self.checkpoint_dir, "evaluate")

        # if tf.train.latest_checkpoint(self.checkpoint_dir):
        #     status = checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
        #     # status.expect_partial()
        #     print("Model restored from checkpoint.")
        # else:
        #     raise ValueError("No checkpoint found. Cannot evaluate without a trained model.")

        total_bleu_case_sensitive = 0
        total_bleu_case_insensitive = 0
        num_examples = 0

        for (batch, (src, tgt_out)) in enumerate(test_dataset):
            target_input = tgt_out[:, :-1]  # Note: We're assuming this matches the approach used during training
            encoder_padding_mask, combined_mask, decoder_padding_mask = get_masks(src, target_input)
            tgt_pred, _, _ = self.model(src, target_input, encoder_padding_mask, combined_mask, decoder_padding_mask,
                                        training=False)
            tgt_pred_tokens = tf.argmax(tgt_pred, axis=-1)
            # print("tgt_pred_token shape", tgt_pred_tokens.shape)
            # print("tgt_out shape", tgt_out.shape)
            for real, pred in zip(tgt_out.numpy(), tgt_pred_tokens):
                # Remove padding tokens or any other special tokens
                real = [token for token in real if token != self.language_processor.pad()[1]]  # Assuming tgt is Amharic
                pred_numpy = pred.numpy()
                pred = [token for token in pred_numpy if token != self.language_processor.pad()[1]]

                # Decode using the appropriate tokenizer based on the target file extension
                if self.language_processor.tgt_file.endswith('.en'):
                    real_tokens = self.language_processor.tokenizer.decode(real, skip_special_tokens=True)
                    pred_tokens = self.language_processor.tokenizer.decode(pred, skip_special_tokens=True)
                elif self.language_processor.tgt_file.endswith('.am'):
                    with open(self.language_processor.tgt_vocab_file, 'r', encoding='utf-8') as f:
                        vocab = {line.split()[0]: int(line.split()[1]) for line in f}
                    inverse_vocab = {v: k for k, v in vocab.items()}
                    real_tokens = ' '.join([inverse_vocab.get(token, '[UNK]') for token in real])
                    pred_tokens = ' '.join([inverse_vocab.get(token, '[UNK]') for token in pred])
                else:
                    raise ValueError(f"Unsupported target file extension in {self.language_processor.tgt_file}")

                case_sensitive_bleu = self.compute_bleu(real_tokens, pred_tokens)
                case_insensitive_bleu = self.compute_bleu(real_tokens.lower(), pred_tokens.lower())

                total_bleu_case_sensitive += case_sensitive_bleu
                total_bleu_case_insensitive += case_insensitive_bleu

                num_examples += 1

        avg_bleu_case_sensitive = total_bleu_case_sensitive / num_examples
        avg_bleu_case_insensitive = total_bleu_case_insensitive / num_examples

        return avg_bleu_case_sensitive, avg_bleu_case_insensitive

    def translate(self, input_text):
        input_text = self.tokenizer_src.encode(input_text, return_tensors="tf")
        input_text = tf.expand_dims(input_text, axis=0)

        enc_padding_mask, combined_mask, dec_padding_mask = get_masks(input_text, input_text)

        output_ids = self.model.greedy_decode(input_text, enc_padding_mask, combined_mask, dec_padding_mask)
        output_text = self.tokenizer_tgt.decode(output_ids[0], skip_special_tokens=True)

        return output_text

    def save_translation(self, input_text, output_file):
        translated_text = self.translate(input_text)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(translated_text)

    def compute_bleu(self, reference, candidate):
        """
        Compute BLEU score between reference and candidate sentences.

        :param reference: str, the ground truth sentence
        :param candidate: str, the predicted sentence
        :return: float, BLEU score
        """
        # Tokenize sentences into words
        reference_tokens = reference.split()
        candidate_tokens = candidate.split()

        # Compute BLEU score
        bleu_score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=SmoothingFunction().method1)
        return bleu_score

# Load your trained model and tokenizer here
# trained_model = ...
# src_lang_data = ...
# tgt_lang_data = ...


# Load your trained model and tokenizer here
# trained_model = ...
# target_tokenizer = ...
#
# # Load your test dataset here
# test_dataset = ...
#
# # Create the evaluator instance
# evaluator = CtnTransformerEvaluator(trained_model, target_tokenizer)
#
# # Evaluate the model and get the BLEU score
# bleu_score = evaluator.evaluate(test_dataset)
# print("BLEU Score:", bleu_score)

# Load the model and tokenizer
# model = ...
# tokenizer = ...
# src_lang_data = ...
# tgt_lang_data = ...

# Load test dataset
# test_dataset = ...

# Create the evaluator
# evaluator = TransformerEvaluator(model, tokenizer, src_lang_data, tgt_lang_data)

# Evaluate the model on test data
# avg_loss = evaluator.evaluate(test_dataset)
# print("Average Loss:", avg_loss.numpy())

# Translate a sentence and save the translation
# input_sentence = "..."
# output_file = "translation.txt"
# evaluator.save_translation(input_sentence, output_file)
