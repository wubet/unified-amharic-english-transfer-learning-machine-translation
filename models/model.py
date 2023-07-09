from abc import ABC

import tensorflow as tf
from tensorflow.keras import layers
from encoders.encoder import Encoder
from decoders.decoder import Decoder
from layers.encoder_layer import EncoderLayer
from layers.decoder_layer import DecoderLayer


class Transformer(tf.keras.Model, ABC):
    """
    Complete transformer with an Encoder and a Decoder
    """

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout_rate=0.1):
        # self.enc_layer = EncoderLayer(d_model, num_heads, dff, dropout_rate)
        # self.decoder_layers = [DecoderLayer(d_model, num_heads, dff, dropout_rate)
        #                        for _ in range(num_layers)]
        # self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, dropout_rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, dropout_rate)
        self.final_layer = layers.Dense(target_vocab_size, activation='softmax')
        # self.final_layer = layers.Dense(target_vocab_size, activation='softmax')

    def call(self, inputs, targets, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        """
         Forward pass for the entire Transformer
        :param inputs:
            Tensor of shape (batch_size, input_seq_len, fully_connected_dim)
                              An array of the indexes of the words in the input sentence
        :param targets:
            Tensor of shape (batch_size, target_seq_len, fully_connected_dim)
                              An array of the indexes of the words in the output sentence
        :param training:
            Boolean, set to true to activate
                        the training mode for dropout layers
        :param enc_padding_mask:
            Boolean mask to ensure that the padding is not
                    treated as part of the input
        :param look_ahead_mask:
            Boolean mask for the target_input
        :param dec_padding_mask:
            Boolean mask for the second multihead attention layer
        :return:
            final_output -- Describe me
            attention_weights - Dictionary of tensors containing all the attention weights for the decoder
                                each of shape Tensor of shape (batch_size, num_heads, target_seq_len, input_seq_len)
        """
        # test
        # enc_output = self.encoder(tf.random.uniform((64, 62)),
        #                           training=False, mask=None)
        # call self.encoder with the appropriate arguments to get the encoder output
        enc_output = self.encoder(inputs, training, enc_padding_mask)
        # call self.decoder with the appropriate arguments to get the decoder output
        # dec_output.shape == (batch_size, tar_seq_len, fully_connected_dim)
        print("enc_output",  enc_output)
        dec_output, attention_weights = self.decoder(targets, enc_output, training, look_ahead_mask, dec_padding_mask)
        # (tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        # pass decoder output through a linear layer and softmax (~2 lines)
        final_output = self.final_layer(dec_output)

        return final_output, attention_weights

    # # create a mask for the padding tokens
    # def create_padding_mask(self, seq):
    #     seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    #     # add extra dimensions so that we can add the padding
    #     # to the attention logits.
    #     return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    def create_padding_mask(self, seq):
        """
        Creates a matrix mask for the padding cells
        :param seq:
            decoder_token_ids -- (n, m) matrix
        :return:
            mask -- (n, 1, 1, m) binary tensor
        """
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, :]  # (batch_size, seq_len)

    # def create_padding_mask(self, seq):
    #     seq = tf.cast(tf.math.equal(seq, 0), tf.bool)
    #     return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    # def create_padding_mask(self, seq):
    #     seq = tf.cast(tf.math.not_equal(seq, 0), tf.bool)
    #     return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    # def create_padding_mask(self, seq):
    #     seq = tf.cast(tf.math.not_equal(seq, 0), tf.bool)
    #     # expanding dimensions of mask to match dimensions of query, key and value tensors
    #     return seq[:, tf.newaxis, :]  # (batch_size, 1, seq_len)

    # used in the decoder to mask future tokens in a sequence during training.
    def create_look_ahead_mask(self, size):
        """
        Returns an upper triangular matrix filled with ones
        :param size:
            sequence_length -- matrix size
        :return:
            mask -- (size, size) tensor
        """
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)

# def positional_encoding(position, d_model):
#     angle_rates = 1 / tf.pow(10000, (2 * (tf.range(d_model, dtype=tf.float32) // 2)) / tf.cast(d_model, tf.float32))
#     angle_rads = tf.reshape(angle_rates, (1, -1))
#     angles = position * angle_rads
#
#     sines = tf.math.sin(angles[:, 0::2])
#     cosines = tf.math.cos(angles[:, 1::2])
#
#     pos_encoding = tf.concat([sines, cosines], axis=-1)
#     pos_encoding = tf.expand_dims(pos_encoding, 0)
#
#     return pos_encoding


# class Encoder(tf.keras.layers.Layer):
#     def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, dropout_rate=0.1):
#         super(Encoder, self).__init__()
#
#         self.d_model = d_model
#         self.num_layers = num_layers
#
#         self.embedding = layers.Embedding(input_vocab_size, d_model)
#         self.pos_encoding = positional_encoding(input_vocab_size, self.d_model)
#
#         self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
#
#         self.dropout = layers.Dropout(dropout_rate)
#
#     def call(self, x, training, mask):
#         seq_len = tf.shape(x)[1]
#         attention_weights = {}
#
#         x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
#         x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
#         x += self.pos_encoding[:, :seq_len, :]
#
#         x = self.dropout(x, training=training)
#
#         for i in range(self.num_layers):
#             x, attention_weight = self.enc_layers[i](x, training, mask)
#             attention_weights[f'encoder_layer{i + 1}'] = attention_weight
#
#         return x, attention_weights


# class EncoderLayer(tf.keras.layers.Layer):
#     def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
#         super(EncoderLayer, self).__init__()
#
#         self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
#         self.ffn = tf.keras.Sequential([
#             layers.Dense(dff, activation='relu'),
#             layers.Dense(d_model)
#         ])
#
#         self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
#         self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
#
#         self.dropout1 = layers.Dropout(dropout_rate)
#         self.dropout2 = layers.Dropout(dropout_rate)


# class DecoderLayer(tf.keras.layers.Layer):
#     def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
#         super(DecoderLayer, self).__init__()
#
#         self.mha1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
#         self.mha2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
#
#         self.ffn = tf.keras.Sequential([
#             layers.Dense(dff, activation='relu'),
#             layers.Dense(d_model)
#         ])
#
#         self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
#         self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
#         self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
#
#         self.dropout1 = layers.Dropout(dropout_rate)
#         self.dropout2 = layers.Dropout(dropout_rate)
#         self.dropout3 = layers.Dropout(dropout_rate)
#
#     def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
#         attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
#         attn1 = self.dropout1(attn1, training=training)
#         out1 = self.layernorm1(attn1 + x)
#
#         attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
#         attn2 = self.dropout2(attn2, training=training)
#         out2 = self.layernorm2(attn2 + out1)
#
#         ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
#         ffn_output = self.dropout3(ffn_output, training=training)
#         out3 = self.layernorm3(ffn_output + out2)
#
#         return out3, attn_weights_block1, attn_weights_block2


# class Decoder(tf.keras.layers.Layer):
#     def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, dropout_rate=0.1):
#         super(Decoder, self).__init__()
#
#         self.d_model = d_model
#         self.num_layers = num_layers
#
#         self.embedding = layers.Embedding(target_vocab_size, d_model)
#         self.pos_encoding = positional_encoding(target_vocab_size, d_model)
#
#         self.dec_layers = [DecoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
#
#         self.dropout = layers.Dropout(dropout_rate)
#
#     def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
#         seq_len = tf.shape(x)[1]
#         attention_weights = {}
#
#         x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
#         x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
#         x += self.pos_encoding[:, :seq_len, :]
#
#         x = self.dropout(x, training=training)
#
#         for i in range(self.num_layers):
#             x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
#             attention_weights[f'decoder_layer{i + 1}_block1'] = block1
#             attention_weights[f'decoder_layer{i + 1}_block2'] = block2
#
#         return x, attention_weights
