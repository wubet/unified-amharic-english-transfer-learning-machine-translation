import tensorflow as tf
from tensorflow.keras import layers


class DecoderLayer(tf.keras.layers.Layer):
    """
    The decoder layer is composed by two multi-head attention blocks,
    one that takes the new input and uses self-attention, and the other
    one that combines it with the output of the encoder, followed by a
    fully connected block.
    """

    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout_rate)
        self.mha2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout_rate)

        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)

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
        # BLOCK 1
        # calculate self-attention and return attention scores as attn_weights_block1.
        # Dropout will be applied during training (~1 line).
        # attn1, attn_weights_block1 = self.mha1(query=x, value=x, key=x,
        #                                        attention_mask=look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1, attn_weights_block1 = self.mha1(query=x, value=x, key=x, attention_mask=look_ahead_mask,
                                               return_attention_scores=True,
                                               training=training)  # (batch_size, target_seq_len, d_model)
        # attn1 = self.dropout1(attn1, training=training)
        # apply layer normalization (layernorm1) to the sum of the attention output and the input (~1 line)
        out1 = self.layernorm1(attn1 + x)

        # BLOCK 2
        # calculate self-attention using the Q from the first block and K and V from the encoder output.
        # Dropout will be applied during training
        # Return attention scores as attn_weights_block2 (~1 line)
        # attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1,
        #                                        padding_mask)  # (batch_size, target_seq_len, d_model)
        # attn2, attn_weights_block2 = self.mha2(query=out1, value=enc_output, key=enc_output,
        #                                        attention_mask=padding_mask)  # (batch_size, target_seq_len, d_model)
        print("enc_output.shape", enc_output.shape)
        attn2, attn_weights_block2 = self.mha2(query=out1, value=enc_output, key=enc_output,
                                               attention_mask=padding_mask,
                                               return_attention_scores=True,
                                               training=training)  # (batch_size, target_seq_len, d_model)
        # attn2 = self.dropout2(attn2, training=training)
        # apply layer normalization (layernorm2) to the sum of the attention output and the output of the first block
        # (~1 line)
        out2 = self.layernorm2(out1 + attn2)

        # BLOCK 3
        # pass the output of the second block through a ffn
        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        # apply a dropout layer to the ffn output
        ffn_output = self.dropout3(ffn_output, training=training)
        # apply layer normalization (layernorm3) to the sum of the ffn output and the output of the second block
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2
