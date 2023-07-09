from abc import ABC

import tensorflow as tf
from tensorflow.keras import layers
from encoders.encoder import Encoder
from decoders.decoder import Decoder
from layers.encoder_layer import EncoderLayer
from layers.decoder_layer import DecoderLayer


class Transformer(tf.keras.Model, ABC):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout_rate=0.1):
        # self.enc_layer = EncoderLayer(d_model, num_heads, dff, dropout_rate)
        # self.decoder_layers = [DecoderLayer(d_model, num_heads, dff, dropout_rate)
        #                        for _ in range(num_layers)]
        # self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        super(Transformer, self).__init__()

        self._encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, dropout_rate)
        self._decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, dropout_rate)
        self._final_layer = layers.Dense(target_vocab_size)

    def call(self, inputs, targets, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self._encoder(inputs, training, enc_padding_mask)
        dec_output, attention_weights = self._decoder(targets, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self._final_layer(dec_output)

        return final_output, attention_weights

    # create a mask for the padding tokens
    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        # add extra dimensions so that we can add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    # used in the decoder to mask future tokens in a sequence during training.
    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)

    # # applies an encoder layer to the inputs
    # def encoder(self, x, mask):
    #     return self.enc_layer(x, mask)
    #
    # # generate the decoder output
    # def decoder(self, x, enc_output, look_ahead_mask, padding_mask):
    #     for i in range(len(self.decoder_layers)):
    #         x, _, _ = self.decoder_layers[i](x, enc_output, look_ahead_mask, padding_mask)
    #     return x  # (batch_size, target_seq_len, d_model)
    #
    # # linear transformation to reshapes the output of the decoder
    # def final_layer(self, x):
    #     return self.final_layer(x)
