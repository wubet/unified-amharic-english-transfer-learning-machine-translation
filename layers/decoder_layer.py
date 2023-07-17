import tensorflow as tf
from tensorflow.keras import layers


class DecoderLayer(tf.keras.layers.Layer):
    """
    The decoder layer is composed by two multi-head attention blocks,
    one that takes the new input and uses self-attention, and the other
    one that combines it with the output of the encoder, followed by a
    fully connected block.
    """

    # Initialize the layer with its internal components.
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        # Call the initializer of the parent class.
        super(DecoderLayer, self).__init__()

        # Define the first multi-head attention layer, this will perform self-attention on the input.
        self.mha1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout_rate)

        # Define the second multi-head attention layer, this will perform attention on the encoder output.
        self.mha2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout_rate)

        # Define the pointwise feed-forward network.
        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])

        # Define layer normalization layers, to stabilize the layer's inputs.
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)

        # Define dropout layers to prevent overfitting.
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dropout3 = layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """
        Forward pass for the Decoder Layer
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
            out3 -- Tensor of shape (batch_size, target_seq_len, fully_connected_dim)
            attn_weights_block1 -- Tensor of shape(batch_size, num_heads, target_seq_len, input_seq_len)
            attn_weights_block2 -- Tensor of shape(batch_size, num_heads, target_seq_len, input_seq_len)
        """
        # Block 1: self-attention with look-ahead mask and layer normalization.
        attn1, attn_weights_block1 = self.mha1(query=x, value=x, key=x, attention_mask=look_ahead_mask,
                                               return_attention_scores=True,
                                               training=training)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        # Block 2: attention with encoder output and padding mask and layer normalization.
        attn2, attn_weights_block2 = self.mha2(query=out1, value=enc_output, key=enc_output,
                                               attention_mask=padding_mask,
                                               return_attention_scores=True,
                                               training=training)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        # Block 3: pointwise feed-forward network with layer normalization.
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        # Return the output and the attention weights.
        return out3, attn_weights_block1, attn_weights_block2
