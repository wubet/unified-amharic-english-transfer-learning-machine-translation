import tensorflow as tf
from tensorflow.keras import layers
from common.utils import positional_encoding
from layers.encoder_layer import EncoderLayer
import numpy as np


class Encoder(tf.keras.layers.Layer):
    """
    The entire Encoder starts by passing the input to an embedding layer
    and using positional encoding to then pass the output through a stack of
    encoder Layers
    """

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, dropout_rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(input_vocab_size, self.d_model)

        # self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]

        self.dropout = layers.Dropout(dropout_rate)

    # def call(self, inputs, padding_mask, training):
    #     """
    #     Computes the output of the encoder stack of layers.
    #
    #     Args:
    #         inputs: float tensor of shape [batch_size, src_seq_len, hidden_size], the input source sequences.
    #         padding_mask: float tensor of shape [batch_size, 1, 1, src_seq_len], populated with either 0 (for tokens to keep) or 1 (for tokens to be masked).
    #         training: bool scalar, True if in training mode.
    #
    #     Returns:
    #         outputs: float tensor of shape [batch_size, src_seq_len, hidden_size], the output source sequences.
    #         attention_weights: a list of float tensors containing attention weights from each layer in the stack.
    #     """
    #     attention_weights = []
    #     for layer in self.enc_layers:
    #         inputs, weights = layer.call(inputs, padding_mask, training)
    #         attention_weights.append(weights)
    #     outputs = self._layernorm(inputs)
    #
    #     return outputs, attention_weights

    # def call(self, x, training, mask):
    #     """
    #      Forward pass for the Encoder
    #     :param x:
    #         Tensor of shape (batch_size, input_seq_len)
    #     :param training:
    #          Boolean, set to true to activate the training mode for dropout layers
    #     :param mask:
    #         Boolean mask to ensure that the padding is not treated as part of the input
    #     :return:
    #         out2 -- Tensor of shape (batch_size, input_seq_len, fully_connected_dim/dff)
    #     """
    #
    #     # mask = create_padding_mask(x)
    #     seq_len = tf.shape(x)[1]
    #     # Pass input through the Embedding layer
    #     x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    #     # Scale embedding by multiplying it by the square root of the embedding dimension/dff
    #     # x *= np.sqrt(self.d_model)
    #     x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    #     # Add the position encoding to embedding
    #     x += self.pos_encoding[:, :seq_len, :]
    #     # Pass the encoded embedding through a dropout layer
    #     x = self.dropout(x, training=training)
    #     attention_weights = []
    #     # Pass the output through the stack of encoding layers
    #     for i in range(self.num_layers):
    #         x, weights = self.enc_layers[i](x, training, mask)
    #         attention_weights.append(weights)
    #
    #     attention_weights = tf.stack(attention_weights)
    #
    #     return x, attention_weights  # (batch_size, input_seq_len, fully_connected_dim/dff)

    # def call(self, x, training, mask):
    #     seq_len = tf.shape(x)[1]
    #
    #     x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    #     x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    #     x += self.pos_encoding[:, :seq_len, :]
    #
    #     x = self.dropout(x, training=training)
    #
    #     attention_weights = tf.TensorArray(dtype=tf.float32, size=self.num_layers)
    #
    #     for i in range(self.num_layers):
    #         x, attention_weight = self.enc_layers[i](x, training, mask)
    #         attention_weights = attention_weights.write(i, attention_weight)
    #
    #     return x, attention_weights.stack()

    @tf.function
    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        print("Encoder layers:", self.enc_layers)
        attention_weights = []
        for layer in self.enc_layers:
            print("Before layer", layer)
            x = layer(x, training, mask)
            print("After layer", layer)
            # attention_weights.append(weights)

        # The last layer's output is used
        return x

    # def call(self, x, training, mask):
    #     seq_len = tf.shape(x)[1]
    #
    #     x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    #     x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    #     x += self.pos_encoding[:, :seq_len, :]
    #
    #     x = self.dropout(x, training=training)
    #
    #     attention_weights = tf.TensorArray(tf.float32, size=self.num_layers)
    #
    #     def map_fn(idx, inputs):
    #         x, attention_weights = inputs
    #         x, weights = self.enc_layers[idx](x, training, mask)
    #         attention_weights = attention_weights.write(idx, weights)
    #         return x, attention_weights
    #
    #     x, attention_weights = tf.map_fn(
    #         map_fn, tf.range(self.num_layers),
    #         dtype=(tf.float32, tf.float32),
    #         fn_output_signature=(
    #             tf.TensorSpec(shape=x.shape, dtype=tf.float32),
    #             tf.TensorSpec(shape=attention_weights.shape, dtype=tf.float32)
    #         ),
    #         elems=(x, attention_weights)
    #     )
