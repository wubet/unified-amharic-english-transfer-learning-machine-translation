import tensorflow as tf
from transformers import BertModel, BertTokenizer
import numpy as np


# class CTNMTModel(tf.keras.Model):
#     def __init__(self, bert_model, nmt_model):
#         super(CTNMTModel, self).__init__()
#         self.bert_model = bert_model
#         self.nmt_model = nmt_model
#         self.alpha = tf.Variable(0.5,
#                                  trainable=False)  # hyper-parameter that balances preference between pre-training distillation and NMT objective
#         self.context_gate = tf.keras.layers.Dense(1, activation='sigmoid')
#
#     def call(self, inputs):
#         # Obtain the BERT hidden states
#         bert_outputs = self.bert_model(inputs)[0]
#         # Obtain the NMT hidden states
#         nmt_outputs = self.nmt_model.encoder(inputs)
#
#         # Asymptotic distillation
#         l2_loss = tf.reduce_mean(tf.math.squared_difference(bert_outputs, nmt_outputs))
#
#         # Dynamic switch
#         g = self.context_gate(tf.concat([bert_outputs, nmt_outputs], axis=-1))
#         switched_output = g * bert_outputs + (1 - g) * nmt_outputs
#
#         return switched_output, l2_loss


class CTNMTransformer(tf.keras.Model):
    def __init__(self, gate_size):
        super().__init__()  # add this line
        # Initialize gate parameters
        self.W = self.add_weight(shape=(gate_size, gate_size), initializer='random_normal', dtype=tf.float32)
        self.U = self.add_weight(shape=(gate_size, gate_size), initializer='random_normal', dtype=tf.float32)
        self.b = self.add_weight(shape=(gate_size,), initializer='zeros', dtype=tf.float32)

        # Initialize the MSE loss function
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        ...

    def dynamic_switch(self, teacher_enc_output, student_enc_output):
        # Calculate the gate value
        g = tf.sigmoid(tf.matmul(teacher_enc_output, self.W) + tf.matmul(student_enc_output, self.U) + self.b)
        # Combine the teacher's and student's outputs
        combined_enc_output = g * teacher_enc_output + (1 - g) * student_enc_output
        return combined_enc_output

    def asymptotic_distillation(self, teacher_hidden_state, student_hidden_state):
        # Calculate the MSE loss
        distillation_loss = self.mse_loss(teacher_hidden_state, student_hidden_state)
        return distillation_loss

    ...
