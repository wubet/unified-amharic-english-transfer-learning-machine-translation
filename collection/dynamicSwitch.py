import tensorflow as tf
from tensorflow.keras.models import Model, Input, Dense
from transformers import TFBertModel


class DynamicSwitchingLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(DynamicSwitchingLayer, self).__init__()

    def call(self, inputs):
        bert_output, nmt_output = inputs
        alpha = tf.nn.sigmoid(self.compute_alpha(bert_output, nmt_output))  # Add your function to compute alpha
        return alpha * bert_output + (1 - alpha) * nmt_output


# Next, we introduce a dynamic switch mechanism that combines the BERT and Transformer's encoded embeddings.
class DynamicSwitch(tf.keras.layers.Layer):
    def __init__(self):
        super(DynamicSwitch, self).__init__()
        self.gate = Dense(1, activation='sigmoid')

    def call(self, bert_emb, nmt_emb):
        switch = self.gate(bert_emb)
        return switch * bert_emb + (1 - switch) * nmt_emb


dynamic_switch = DynamicSwitch()


# Dynamic Switch for Knowledge Fusion:
class DynamicSwitch(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gate = Dense(1, activation='sigmoid')

    def call(self, bert_emb, nmt_emb):
        switch = self.gate(bert_emb)
        return switch * bert_emb + (1 - switch) * nmt_emb
