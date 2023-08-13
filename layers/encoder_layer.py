import tensorflow as tf
# from tensorflow.keras import layers
from keras import layers
from common.attention import *


class EncoderLayer(tf.keras.layers.Layer):
    # Initialize the layer with its internal components.
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        """
        Initialize the EncoderLayer class with the following parameters:

       :param d_model: int, the dimension of the key, query, and value vectors, commonly referred to as the embedding dimension.
       :param num_heads: int, the number of attention heads for the multi-head attention mechanism.
       :param dff: int, the dimensionality of the "feed forward" network inside the encoder layer.
       :param dropout_rate: float, the dropout rate to be applied to certain layers to prevent overfitting. Default is 0.1.

       :return: None
       """
        # Call the initializer of the parent class.
        super(EncoderLayer, self).__init__()

        # Define the multi-head attention layer.
        # self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout_rate)
        self.mha = MultiHeadAttention(d_model, num_heads)

        # Define the pointwise feed-forward network.
        self.ffn = PositionWiseFFN(d_model, dff)

        # Define layer normalization layers to stabilize the layer's inputs.
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        # Define dropout layers to prevent overfitting.
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        """
        Forward pass for the Encoder Layer
        :param x:
            x -- Tensor of shape (batch_size, input_seq_len, fully_connected_dim)
        :param training:
            Boolean, set to true to activate the training mode for dropout layers
        :param mask:
            Boolean mask to ensure that the padding is not treated as part of the input
        :return:
            encoder_layer_out -- Tensor of shape (batch_size, input_seq_len, fully_connected_dim)
            attn_weights -- attention weights from MultiHeadAttention layer
        """
        # Calculate self-attention and apply dropout.
        # attn_output = self.mha(query=x, value=x, key=x, attention_mask=mask, training=training)
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)

        # Apply layer normalization to the sum of the input and the attention output.
        out1 = self.layernorm1(x + attn_output)

        # Pass the output of the multi-head attention layer through the feed-forward network and apply dropout.
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training)

        # Apply layer normalization to the sum of the output from the first block and the feed-forward network output.
        encoder_layer_out = self.layernorm2(out1 + ffn_output)

        # Return the output of the encoder layer.
        return encoder_layer_out