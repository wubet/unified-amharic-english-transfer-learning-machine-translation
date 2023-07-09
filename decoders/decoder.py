import tensorflow as tf
from tensorflow.keras import layers
from common.utils import positional_encoding
from layers.decoder_layer import DecoderLayer
import numpy as np


class Decoder(tf.keras.layers.Layer):
    """
    The entire Encoder is starts by passing the target input to an embedding layer
    and using positional encoding to then pass the output through a stack of
    decoder Layers
    """
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(target_vocab_size, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]

        self.dropout = layers.Dropout(dropout_rate)

    @tf.function
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """
        Forward  pass for the Decoder
        :param x:
            Tensor of shape (batch_size, target_seq_len, fully_connected_dim)
        :param enc_output:
            Tensor of shape(batch_size, input_seq_len, fully_connected_dim)
        :param training:
            Boolean, set to true to activate the training mode for dropout layers
        :param look_ahead_mask:
            Boolean mask for the target_input
        :param padding_mask:
            Boolean mask for the second multihead attention layer
        :return:
            x -- Tensor of shape (batch_size, target_seq_len, fully_connected_dim)
            attention_weights - Dictionary of tensors containing all the attention weights
                                each of shape Tensor of shape (batch_size, num_heads, target_seq_len, input_seq_len)
        """
        # seq_len = tf.shape(x)[1]
        # attention_weights = {}
        #
        # # create word embeddings
        # x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        # # scale embeddings by multiplying by the square root of their dimension
        # # x *= np.sqrt(self.d_model)
        # x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # # calculate positional encodings and add to word embedding
        # x += self.pos_encoding[:, :seq_len, :]
        # # apply a dropout layer to x
        # x = self.dropout(x, training=training)
        # # use a for loop to pass x through a stack of decoder layers and update attention_weights (~4 lines total)
        # for i in range(self.num_layers):
        #     # pass x and the encoder output through a stack of decoder layers and save the attention weights
        #     # of block 1 and 2 (~1 line)
        #     x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
        #     attention_weights[f'decoder_layer{i + 1}_block1'] = block1
        #     attention_weights[f'decoder_layer{i + 1}_block2'] = block2
        #
        # # x.shape == (batch_size, target_seq_len, fully_connected_dim/dff)
        # return x, attention_weights
        # create word embeddings
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        # scale embeddings by multiplying by the square root of their dimension
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # calculate positional encodings and add to word embedding
        x += self.pos_encoding[:, :seq_len, :]
        # apply a dropout layer to x
        x = self.dropout(x, training=training)

        attention_weights = {}
        for i, dec_layer in enumerate(self.dec_layers):
            # pass x and the encoder output through a stack of decoder layers and save the attention weights
            # of block 1 and 2
            x, block1, block2 = dec_layer(x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights[f'decoder_layer{i + 1}_block1'] = block1
            attention_weights[f'decoder_layer{i + 1}_block2'] = block2

        # x.shape == (batch_size, target_seq_len, fully_connected_dim/dff)
        return x, attention_weights
