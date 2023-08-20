import tensorflow as tf
# from transformers import BertModel, BertTokenizer
import numpy as np


class CTNMTransformer(tf.keras.Model):
    """
        The CTNMTransformer (Curriculum Teacher-Student Neural Machine Translation Transformer) model is designed to
        integrate both teacher and student model outputs dynamically and also facilitate knowledge distillation
        between the teacher and student models during training.

        Attributes:
        ----------
        W, U : tf.Variable
            Weight matrices for calculating the gate value.
        b : tf.Variable
            Bias term for calculating the gate value.
        mse_loss : tf.keras.losses.MeanSquaredError
            Mean squared error loss instance for computing distillation loss.
    """

    def __init__(self, gate_size):
        """
            Initializes the CTNMTransformer with given gate size.

            Parameters:
            ----------
            gate_size : int
                The size for the gate matrix which determines the gate value during dynamic switching.
        """
        super().__init__()  # add this line
        # Initialize gate parameters
        self.W = self.add_weight(shape=(gate_size, gate_size), initializer='random_normal', dtype=tf.float32)
        self.U = self.add_weight(shape=(gate_size, gate_size), initializer='random_normal', dtype=tf.float32)
        self.b = self.add_weight(shape=(gate_size,), initializer='zeros', dtype=tf.float32)

        # Initialize the MSE loss function
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        ...

    def dynamic_switch(self, teacher_enc_output, student_enc_output):
        """
            Perform dynamic switching to integrate teacher and student encoder outputs.

            Parameters:
            ----------
            teacher_enc_output : tf.Tensor
                The output from the teacher's encoder.
            student_enc_output : tf.Tensor
                The output from the student's encoder.

            Returns:
            ----------
            tf.Tensor:
                The combined encoder output using a dynamic gate mechanism.
        """
        # Calculate the gate value
        g = tf.sigmoid(tf.matmul(teacher_enc_output, self.W) + tf.matmul(student_enc_output, self.U) + self.b)
        # Combine the teacher's and student's outputs
        combined_enc_output = g * teacher_enc_output + (1 - g) * student_enc_output
        return combined_enc_output

    def asymptotic_distillation(self, teacher_hidden_state, student_hidden_state):
        """
            Computes the Mean Squared Error (MSE) between the teacher and student hidden states
            as a measure of distillation loss.

            Parameters:
            ----------
            teacher_hidden_state : tf.Tensor
                Hidden state from the teacher's model.
            student_hidden_state : tf.Tensor
                Hidden state from the student's model.

            Returns:
            ----------
            tf.Tensor:
                The computed MSE loss between teacher and student hidden states.
        """
        # Calculate the MSE loss
        distillation_loss = self.mse_loss(teacher_hidden_state, student_hidden_state)
        return distillation_loss

    def generate_distillation_rate(self, current_epoch, total_epochs, start_rate=1.0, end_rate=0.0, constant_epochs=5):
        """
            Linearly decrease the distillation_rate from start_rate to end_rate over the total_epochs.

            Parameters:
            ----------
            current_epoch : int
                The current epoch number (starting from 0).
            total_epochs : int
                The total number of epochs during training.
            start_rate : float, optional
                The starting value for distillation_rate at the beginning (epoch 0). Defaults to 1.0.
            end_rate : float, optional
                The ending value for distillation_rate at the end of training. Defaults to 0.0.

            Returns:
            ----------
            float:
                The calculated distillation_rate for the current epoch.
        """
        if current_epoch < constant_epochs:
            return tf.constant(start_rate, dtype=tf.float32)
        adjusted_epochs = total_epochs - constant_epochs
        adjusted_epoch = current_epoch - constant_epochs
        distillation_rate = start_rate - (adjusted_epoch / (adjusted_epochs - 1)) * (start_rate - end_rate)
        return tf.cast(distillation_rate, tf.float32)
