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

    # The constructor for Encoder class.
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, dropout_rate=0.1):
        """
       Initialize the Encoder class with the following parameters:

       :param num_layers: int, the number of encoder layers.
       :param d_model: int, the dimension of the key, query, and value vectors, also known as the embedding dimension.
       :param num_heads: int, the number of attention heads for the multi-head attention mechanism.
       :param dff: int, the dimensionality of the "feed forward" network inside the encoder layer.
       :param input_vocab_size: int, the size of the input vocabulary, used for the embedding layer.
       :param dropout_rate: float, optional, the dropout rate to be applied to certain layers to prevent overfitting. Default is 0.1.
       """
        # Inherit all the properties from the parent class.
        super(Encoder, self).__init__()

        # Initialize class variables with the parameters passed in the constructor.
        self.d_model = d_model
        self.num_layers = num_layers

        # Create an embedding layer for the input sequence.
        self.embedding = layers.Embedding(input_vocab_size, d_model)

        # Generate the positional encodings.
        self.pos_encoding = positional_encoding(input_vocab_size, self.d_model)

        # Create a list of EncoderLayer objects.
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]

        # Initialize a dropout layer.
        self.dropout = layers.Dropout(dropout_rate)

        # Forward pass for the Encoder.

    def call(self, x, training, mask):
        """
       Forward pass for the Encoder.

       :param x: Tensor, shape (batch_size, input_seq_len, d_model), the input sequence.
       :param training: Boolean, set to true to activate the training mode for dropout layers.
       :param mask: Boolean mask to ensure that padding is not treated as part of the input.

       :return: Tensor, shape (batch_size, input_seq_len, d_model), the output of the encoder.
       """
        # Get the sequence length.
        seq_len = tf.shape(x)[1]

        # Pass the input through the embedding layer.
        x = self.embedding(x)

        # Scale the embeddings by multiplying by the square root of their dimension.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # Add the positional encodings to the embeddings.
        x += self.pos_encoding[:, :seq_len, :]

        # Apply dropout to the embeddings + positional encodings.
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            # Pass the input through each encoder layer.
            x = self.enc_layers[i](x, training, mask)

        return x
