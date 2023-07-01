import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Embedding, LayerNormalization
from tensorflow.keras.models import Model



class Transformer(Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
        # Your code here
        pass

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        # Your code here
        pass