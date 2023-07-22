from common.utils import *


# The class for training the transformer model.
class TrainCntModel:
    # Initialization method that takes the transformer model to be trained as an argument.
    def __init__(self, cnt_transformer):
        # The transformer model to be trained is stored as an instance variable.
        self.cnt_transformer = cnt_transformer

    # The training step method that takes the source and target language input ids.
    def train_step(self, source_language_input_ids, target_language_input_ids):
        print("source_language_input_ids: ", source_language_input_ids.shape)
        print("target_language_input_ids: ", target_language_input_ids.shape)
        # The target input and real target sequences are extracted from the target language input ids.
        tar_input = target_language_input_ids[:, :-1]
        tar_real = target_language_input_ids[:, 1:]

        # GradientTape is used for automatic differentiation i.e., to compute the gradient for backpropagation.
        with tf.GradientTape() as tape:
            # The transformer model is used to make predictions on the input sequences.
            # Perform forward pass with student model

            print("tar_input: ", tar_input.shape)
            student_predictions = self.cnt_transformer.transformer_model([source_language_input_ids,
                                                                          tar_input], training=True)

            print("predictions: ", student_predictions.shape)
            # If a teacher model path is provided and the teacher transformer model is defined,
            # it's used for the Knowledge Distillation approach.
            if self.cnt_transformer.teacher_model_path and self.cnt_transformer.teacher_model:
                # Perform forward pass with teacher model
                teacher_predictions = self.cnt_transformer.teacher_model(source_language_input_ids, training=False)

                # Assuming teacher_predictions and student_predictions are already computed
                self.cnt_transformer.validate_teacher_predictions(teacher_predictions, student_predictions)

                # print(" teacher_predictions",  teacher_predictions.shape)

                # Compute the distillation loss using Asymptotic Distillation (AD)
                distillation_loss = self.cnt_transformer.asymptotic_distillation_loss(target_language_input_ids,
                                                                                      student_predictions,
                                                                                      teacher_predictions)

                # Compute the regular cross-entropy loss
                cross_entropy_loss = self.cnt_transformer.cross_entropy_loss(target_language_input_ids,
                                                                             student_predictions)

                # Combine the losses with a weight factor
                alpha = 0.1  # Weight factor for distillation loss
                total_loss = alpha * distillation_loss + (1 - alpha) * cross_entropy_loss

                # Compute the hidden states of the teacher model and student model
                h_lm = self.cnt_transformer.teacher_model.get_hidden_states(source_language_input_ids)
                h_nmt = self.cnt_transformer.transformer_model.get_hidden_states(source_language_input_ids)

                # Apply the dynamic switching gate
                fused_representation = self.cnt_transformer.dynamic_switching_gate(source_language_input_ids,
                                                                                   h_lm,
                                                                                   h_nmt)
                # Feed the fused representation to the student model
                student_predictions = self.cnt_transformer.transformer_model(fused_representation, training=True)

                # Compute the new cross-entropy loss with the fused representation
                fused_cross_entropy_loss = self.cnt_transformer.cross_entropy_loss(target_language_input_ids,
                                                                                   student_predictions)

                # Add the fused cross-entropy loss to the total loss
                total_loss += fused_cross_entropy_loss
                # The loss is recorded for monitoring.

                self.cnt_transformer.train_loss(total_loss)

                # The training accuracy is calculated for monitoring.
                self.cnt_transformer.train_accuracy(tar_real, student_predictions)
            else:
                loss = self.cnt_transformer.loss_object(tar_real, student_predictions)
                # The loss is recorded for monitoring.
                self.cnt_transformer.train_loss(loss)

                # The training accuracy is calculated for monitoring.
                self.cnt_transformer.train_accuracy(tar_real, student_predictions)

        # The gradients for the trainable parameters are computed using the computed loss.
        gradients = tape.gradient(loss, self.cnt_transformer.trainable_variables)
        # The computed gradients are applied to the trainable variables using the optimizer.
        self.cnt_transformer.optimizer.apply_gradients(
            zip(gradients, self.cnt_transformer.trainable_variables))

    # The training method that takes the dataset and number of epochs as arguments.
    def train(self, dataset, epochs):
        # For each epoch,
        for epoch in range(epochs):
            # The recorded loss and accuracy are reset.
            self.cnt_transformer.train_loss.reset_states()
            self.cnt_transformer.train_accuracy.reset_states()

            # For each batch in the dataset,
            for batch in dataset:
                # The source and target language input ids are extracted.
                (source_language_input_ids, target_language_input_ids) = batch
                # A training step is performed.
                self.train_step(source_language_input_ids, target_language_input_ids)

            # The loss and accuracy for the epoch are printed.
            print(
                f'Epoch {epoch + 1} Loss {self.cnt_transformer.train_loss.result():.4f} Accuracy '
                f'{self.cnt_transformer.train_accuracy.result():.4f}')

            # Rate-scheduled updating - adjust learning rate
            self.cnt_transformer.adjust_learning_rate(epoch)

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

