import tensorflow as tf
from keras.losses import SparseCategoricalCrossentropy
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm


class CtnTransformerEvaluator:
    def __init__(self, model, tokenizer, src_lang_data, tgt_lang_data):
        self.model = model
        self.tokenizer = tokenizer
        self.src_lang_data = src_lang_data
        self.tgt_lang_data = tgt_lang_data

    def evaluate(self, test_dataset):
        references = []
        hypotheses = []

        for src, tgt_inp, tgt_real in tqdm(test_dataset, desc="Evaluating"):
            tgt_pred = self.model.predict(src, tgt_inp)
            tgt_pred_ids = tf.argmax(tgt_pred, axis=-1)

            for real, pred in zip(tgt_real.numpy(), tgt_pred_ids.numpy()):
                real_tokens = self.target_tokenizer.decode(real, skip_special_tokens=True)
                pred_tokens = self.target_tokenizer.decode(pred, skip_special_tokens=True)

                references.append([real_tokens.split()])
                hypotheses.append(pred_tokens.split())

        case_insensitive_score = corpus_bleu(references, hypotheses,
                                             smoothing_function=SmoothingFunction().method4)
        case_sensitive_score = corpus_bleu(references, hypotheses)

        return case_insensitive_score, case_sensitive_score

    def create_masks(self, inp, tar):
        enc_padding_mask = self.create_padding_mask(inp)
        dec_padding_mask = self.create_padding_mask(inp)

        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)

        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask

    def create_padding_mask(self, seq):
        return tf.cast(tf.math.equal(seq, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

    def translate(self, input_text):
        input_text = self.tokenizer.encode(input_text, return_tensors="tf")
        input_text = tf.expand_dims(input_text, axis=0)

        enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(input_text, input_text)

        output_ids = self.model.greedy_decode(input_text, enc_padding_mask, combined_mask, dec_padding_mask)
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return output_text

    def save_translation(self, input_text, output_file):
        translated_text = self.translate(input_text)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(translated_text)


# Load your trained model and tokenizer here
trained_model = ...
target_tokenizer = ...

# Load your test dataset here
test_dataset = ...

# Create the evaluator instance
evaluator = CtnTransformerEvaluator(trained_model, target_tokenizer)

# Evaluate the model and get the BLEU score
bleu_score = evaluator.evaluate(test_dataset)
print("BLEU Score:", bleu_score)


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

