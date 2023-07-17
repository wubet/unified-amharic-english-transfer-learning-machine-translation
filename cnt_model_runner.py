from common.utils import *


# The class for training the transformer model.
class TrainCntModel:
    # Initialization method that takes the transformer model to be trained as an argument.
    def __init__(self, cnt_transformer):
        # The transformer model to be trained is stored as an instance variable.
        self.cnt_transformer = cnt_transformer

    # The training step method that takes the source and target language input ids.
    def train_step(self, source_language_input_ids, target_language_input_ids):
        # The target input and real target sequences are extracted from the target language input ids.
        tar_input = target_language_input_ids[:, :-1]
        tar_real = target_language_input_ids[:, 1:]

        # GradientTape is used for automatic differentiation i.e., to compute the gradient for backpropagation.
        with tf.GradientTape() as tape:
            # The transformer model is used to make predictions on the input sequences.
            predictions = self.cnt_transformer.transformer_model([source_language_input_ids,
                                                                  tar_input], training=True)
            # If a teacher model path is provided and the teacher transformer model is defined,
            # it's used for the Knowledge Distillation approach.
            if self.cnt_transformer.teacher_model_path and self.cnt_transformer.teacher_transformer:
                # The teacher model is used to make predictions.
                teacher_predictions = self.cnt_transformer.teacher_transformer([source_language_input_ids, tar_input],
                                                                               training=False)[0]
                # The distillation loss between the student predictions and teacher predictions is computed.
                loss = self.cnt_transformer.distillation_loss(tar_real, predictions,
                                                              teacher_predictions)
            # If no teacher model is defined, the traditional loss between the real target and predictions is computed.
            else:
                loss = self.cnt_transformer.loss_object(tar_real, predictions)

        # The gradients for the trainable parameters are computed using the computed loss.
        gradients = tape.gradient(loss, self.cnt_transformer.trainable_variables)
        # The computed gradients are applied to the trainable variables using the optimizer.
        self.cnt_transformer.optimizer.apply_gradients(
            zip(gradients, self.cnt_transformer.trainable_variables))

        # The loss is recorded for monitoring.
        self.cnt_transformer.train_loss(loss)
        # The training accuracy is calculated for monitoring.
        self.cnt_transformer.train_accuracy(tar_real, predictions)

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
