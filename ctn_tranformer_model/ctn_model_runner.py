import tensorflow as tf
from common.utils import *
from ctn_tranformer_model.ctnmt import CTNMTransformer
from common.rate_scheduler import RateScheduledOptimizers
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# The class for training the transformer model.
class TrainCntModel:
    # Initialization method that takes the transformer model to be trained as an argument.
    def __init__(self, ctn_transformer):
        # The transformer model to be trained is stored as an instance variable.
        self.ctn_transformer = ctn_transformer

    # @tf.function
    def train_step(self, src, tgt, optimizer, distillation_rate):
        tgt_inp = tgt[:, :-1]
        tgt_real = tgt[:, 1:]

        enc_padding_mask = create_padding_mask(src)

        # Get the tgt shape of the tensor
        tgt_shape = tf.shape(tgt_inp)

        # Extract the value of seq_len
        tgt_seq_len = tgt_shape[1]
        look_ahead_mask = create_look_ahead_attention_mask(tgt_seq_len)

        nmt_mask = create_padding_mask(src)

        teacher_hidden_enc = None

        with tf.GradientTape() as tape:
            teacher_enc_output = self.ctn_transformer.teacher_model(src, training=False)

            # Assuming teacher_enc_output is a dictionary or dictionary-like object
            if 'last_hidden_state' in teacher_enc_output:
                teacher_hidden_enc = teacher_enc_output['last_hidden_state']

            # Get the shape of teacher_hidden_state
            teacher_hidden_shape = tf.shape(teacher_hidden_enc)

            # Extract the size value of the third dimension (weight)
            weight_size = teacher_hidden_shape[2]

            # encoder_inputs = self.ctn_transformer.encoder_input_layer(src)

            # enc_padding_mask_input = self.ctn_transformer.enc_padding_mask_layer(enc_padding_mask)

            # Call the student_encoder with the appropriate inputs.
            student_enc_output = self.ctn_transformer.get_student_encoder()([src, enc_padding_mask], training=True)

            # student_enc_output = self.ctn_transformer.student_encoder([src], training=True)

            # Get the final hidden state for each sequence in the batch
            student_enc_hidden = student_enc_output[:, -1, :]

            ctnm_transformer = CTNMTransformer(weight_size)

            # Apply dynamic switch and asymptotic distillation
            combined_enc_output = ctnm_transformer.dynamic_switch(teacher_hidden_enc, student_enc_output)

            # Use combined_enc_output as input to the student_decoder
            # student_dec_output, _ = self.ctn_transformer.student_decoder(tgt_inp, student_enc_output,
            #                                                              combined_enc_output,
            #                                                              training=True, mask=nmt_mask)
            print("before student_dec_output")
            # student_dec_output = self.ctn_transformer.student_decoder(tgt_inp, look_ahead_mask, nmt_mask, combined_enc_output)
            print(
                f'Calling student_decoder with inputs: {tgt_inp.shape}, {look_ahead_mask.shape}, {nmt_mask.shape}, {combined_enc_output.shape}')
            # student_dec_output = self.ctn_transformer.student_decoder(
            #     [tgt_inp, look_ahead_mask, nmt_mask, combined_enc_output])
            student_dec_output, student_prediction = self.ctn_transformer.get_student_decoder()(
                [tgt_inp, look_ahead_mask, nmt_mask, combined_enc_output])

            print("student_dec_output: ", student_dec_output.shape)
            # print("student_prediction: ", student_prediction.shape)

            # predicted_tokens = tf.argmax(student_dec_output, axis=-1)
            # print("predicted_tokens: ", predicted_tokens.shape)
            # predicted_tokens = tf.argmax(student_dec_output, axis=-1)
            #
            # tgt_real = tf.reshape(tgt_real, [10, 441])
            # print("tgt_real: ", tgt_real.shape)

            nmt_loss = self.ctn_transformer.loss_function(tgt_real, student_prediction)
            # nmt_loss = self.ctn_transformer.loss_function(tgt_real, combined_dec_output)
            print("nmt_loss succeed")

            print("teacher_hidden_enc shape:", teacher_hidden_enc.shape)
            print("student_enc_output shape:", student_enc_output.shape)

            distillation_loss = mse_loss(teacher_hidden_enc, student_enc_output)
            print("distillation_loss succeed")

            combined_loss = nmt_loss * distillation_rate + distillation_loss * (1 - distillation_rate)
            print("combined_loss succeed")

        # gradients = tape.gradient(combined_loss,
        #                           self.ctn_transformer.transformer.encoder.trainable_variables +
        #                           self.ctn_transformer.transformer.decoder.trainable_variables)
        # print("tape.gradient succeed")

            # combined_loss = nmt_loss * distillation_rate + distillation_loss * (1 - distillation_rate)
            # print("combined_loss succeed")

        # Compute gradients for NMT outside the with-block
        nmt_gradients = tape.gradient(combined_loss, self.ctn_transformer.transformer.encoder.trainable_variables +
                                      self.ctn_transformer.transformer.decoder.trainable_variables)
        # print("nmt_gradients:", nmt_gradients)

        # Get NMT variables
        nmt_variables = self.ctn_transformer.transformer.encoder.trainable_variables + self.ctn_transformer.transformer.decoder.trainable_variables
        # print("nmt_variables:", nmt_variables)

        # Apply gradients
        optimizer.apply_gradients(nmt_gradients, nmt_variables)

        return combined_loss, nmt_loss, distillation_loss

    def train(self, dataset, epochs):
        optimizers = RateScheduledOptimizers(learning_rate_nmt=1e-3)
        # For each epoch,
        for epoch in range(epochs):
            # The recorded loss and accuracy are reset.
            self.ctn_transformer.train_loss.reset_states()
            self.ctn_transformer.train_accuracy.reset_states()

            # For each batch in the dataset,
            for batch in dataset:
                # The source and target language input ids are extracted.
                (source_language_input_ids, target_language_input_ids) = batch
                # A training step is performed.
                combined_loss, nmt_loss, distillation_loss = self.train_step(source_language_input_ids,
                                                                             target_language_input_ids,
                                                                             optimizers, 0.5)

            # The loss and accuracy for the epoch are printed.
            print(
                f'Epoch {epoch + 1} Loss {self.ctn_transformer.train_loss.result():.4f} Accuracy '
                f'{self.ctn_transformer.train_accuracy.result():.4f}')

            # Rate-scheduled updating - adjust learning rate
            self.ctn_transformer.adjust_learning_rate(epoch)

    # # The training step method that takes the source and target language input ids.
    # def train_step(self, source_language_input_ids, target_language_input_ids, optimizers):
    #     ct_nmt_model = CTNMTModel(self.ctn_transformer.teacher_model, self.ctn_transformer.transformer_model)
    #     with tf.GradientTape() as tape_bert, tf.GradientTape() as tape_nmt:
    #         switched_output, l2_loss = ct_nmt_model(source_language_input_ids)
    #         nmt_decoded_output = ct_nmt_model.ctn_transformer.transformer_model.decoder(switched_output)
    #
    #         # Cross-entropy loss
    #         ce_loss = tf.keras.losses.sparse_categorical_crossentropy(target_language_input_ids, nmt_decoded_output)
    #
    #         # Total loss
    #         total_loss = ct_nmt_model.alpha * ce_loss + (1 - ct_nmt_model.alpha) * l2_loss
    #
    #     # Calculate and apply gradients
    #     bert_gradients = tape_bert.gradient(total_loss, ct_nmt_model.bert_model.trainable_variables)
    #     nmt_gradients = tape_nmt.gradient(total_loss, self.ctn_transformer.teacher_model.trainable_variables)
    #     optimizers.apply_gradients(bert_gradients, nmt_gradients, self.ctn_transformer.teacher_model
    #                                .trainable_variables, self.ctn_transformer.transformer_model.nmt_model
    #                                .trainable_variables)
    #
    #     return total_loss

    # print("source_language_input_ids: ", source_language_input_ids.shape)
    # print("target_language_input_ids: ", target_language_input_ids.shape)
    # # The target input and real target sequences are extracted from the target language input ids.
    # tar_input = target_language_input_ids[:, :-1]
    # tar_real = target_language_input_ids[:, 1:]
    #
    # # GradientTape is used for automatic differentiation i.e., to compute the gradient for backpropagation.
    # with tf.GradientTape() as tape:
    #     # The transformer model is used to make predictions on the input sequences.
    #     # Perform forward pass with student model
    #
    #     print("tar_input: ", tar_input.shape)
    #     student_predictions = self.cnt_transformer.transformer_model([source_language_input_ids,
    #                                                                   tar_input], training=True)
    #
    #     print("predictions: ", student_predictions.shape)
    #     # If a teacher model path is provided and the teacher transformer model is defined,
    #     # it's used for the Knowledge Distillation approach.
    #     if self.cnt_transformer.teacher_model_path and self.cnt_transformer.teacher_model:
    #         # Perform forward pass with teacher model
    #         teacher_predictions = self.cnt_transformer.teacher_model(source_language_input_ids, training=False)
    #
    #         # Assuming teacher_predictions and student_predictions are already computed
    #         self.cnt_transformer.validate_teacher_predictions(teacher_predictions, student_predictions)
    #
    #         # print(" teacher_predictions",  teacher_predictions.shape)
    #
    #         # Compute the distillation loss using Asymptotic Distillation (AD)
    #         distillation_loss = self.cnt_transformer.asymptotic_distillation_loss(target_language_input_ids,
    #                                                                               student_predictions,
    #                                                                               teacher_predictions)
    #
    #         # Compute the regular cross-entropy loss
    #         cross_entropy_loss = self.cnt_transformer.cross_entropy_loss(target_language_input_ids,
    #                                                                      student_predictions)
    #
    #         # Combine the losses with a weight factor
    #         alpha = 0.1  # Weight factor for distillation loss
    #         total_loss = alpha * distillation_loss + (1 - alpha) * cross_entropy_loss
    #
    #         # Compute the hidden states of the teacher model and student model
    #         h_lm = self.cnt_transformer.teacher_model.get_hidden_states(source_language_input_ids)
    #         h_nmt = self.cnt_transformer.transformer_model.get_hidden_states(source_language_input_ids)
    #
    #         # Apply the dynamic switching gate
    #         fused_representation = self.cnt_transformer.dynamic_switching_gate(source_language_input_ids,
    #                                                                            h_lm,
    #                                                                            h_nmt)
    #         # Feed the fused representation to the student model
    #         student_predictions = self.cnt_transformer.transformer_model(fused_representation, training=True)
    #
    #         # Compute the new cross-entropy loss with the fused representation
    #         fused_cross_entropy_loss = self.cnt_transformer.cross_entropy_loss(target_language_input_ids,
    #                                                                            student_predictions)
    #
    #         # Add the fused cross-entropy loss to the total loss
    #         total_loss += fused_cross_entropy_loss
    #         # The loss is recorded for monitoring.
    #
    #         self.cnt_transformer.train_loss(total_loss)
    #
    #         # The training accuracy is calculated for monitoring.
    #         self.cnt_transformer.train_accuracy(tar_real, student_predictions)
    #     else:
    #         loss = self.cnt_transformer.loss_object(tar_real, student_predictions)
    #         # The loss is recorded for monitoring.
    #         self.cnt_transformer.train_loss(loss)
    #
    #         # The training accuracy is calculated for monitoring.
    #         self.cnt_transformer.train_accuracy(tar_real, student_predictions)
    #
    # # The gradients for the trainable parameters are computed using the computed loss.
    # gradients = tape.gradient(loss, self.cnt_transformer.trainable_variables)
    # # The computed gradients are applied to the trainable variables using the optimizer.
    # self.cnt_transformer.optimizer.apply_gradients(
    #     zip(gradients, self.cnt_transformer.trainable_variables))

    # The training method that takes the dataset and number of epochs as arguments.

    # # Feed the fused representation to the student model
    # student_predictions = self.cnt_transformer.transformer_model(fused_representation, training=True)
    #
    # # Compute the distillation loss using Asymptotic Distillation (AD) with new predictions
    # distillation_loss = self.cnt_transformer.asymptotic_distillation_loss(target_language_input_ids,
    #                                                                       student_predictions,
    #                                                                       teacher_predictions)
    #
    # # Compute the regular cross-entropy loss with new predictions
    # cross_entropy_loss = self.cnt_transformer.cross_entropy_loss(target_language_input_ids,
    #                                                              student_predictions)
    #
    # # Combine the losses with a weight factor
    # alpha = 0.1  # Weight factor for distillation loss
    # total_loss = alpha * distillation_loss + (1 - alpha) * cross_entropy_loss
    #
    # # The loss is recorded for monitoring.
    # self.cnt_transformer.train_loss(total_loss)

    # # Define the ranges of values to test
    # distillation_rate_values = np.linspace(0.1, 1.0, 10)
    # dynamic_switch_rate_values = np.linspace(0.1, 1.0, 10)
    #
    # best_validation_performance = float('inf')
    # best_hyperparameters = None
    #
    # # Iterate over all combinations of hyperparameter values
    # for distillation_rate in distillation_rate_values:
    #     for dynamic_switch_rate in dynamic_switch_rate_values:
    #         # Update the model's hyperparameters
    #         model.distillation_rate = distillation_rate
    #         model.dynamic_switch_rate = dynamic_switch_rate
    #
    #         # Train the model and compute validation performance
    #         model.train(...)
    #         validation_performance = model.validate(...)
    #
    #         # If this is the best validation performance we've seen so far, save these hyperparameters
    #         if validation_performance < best_validation_performance:
    #             best_validation_performance = validation_performance
    #             best_hyperparameters = (distillation_rate, dynamic_switch_rate)
    #
    # print('Best hyperparameters:', best_hyperparameters)
