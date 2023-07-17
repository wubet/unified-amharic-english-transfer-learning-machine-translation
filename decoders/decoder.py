import tensorflow as tf
from tensorflow.keras import layers
from common.utils import positional_encoding
from layers.decoder_layer import DecoderLayer


class Decoder(tf.keras.layers.Layer):
    """
    The entire Encoder is starts by passing the target input to an embedding layer
    and using positional encoding to then pass the output through a stack of
    decoder Layers
    """
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, dropout_rate=0.1):
        # Inherit all the properties from the parent class.
        super(Decoder, self).__init__()

        # Initialize class variables with the parameters passed in the constructor.
        self.d_model = d_model
        self.num_layers = num_layers

        # Create an embedding layer for the input sequence.
        self.embedding = layers.Embedding(target_vocab_size, d_model)

        # Generate the positional encodings.
        self.pos_encoding = positional_encoding(target_vocab_size, d_model)

        # Create a list of DecoderLayer objects.
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]

        # Initialize a dropout layer.
        self.dropout = layers.Dropout(dropout_rate)

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

        # Initialize a dictionary to store the attention weights for each decoder layer.
        attention_weights = {}

        for i in range(self.num_layers):
            # Pass the input through each decoder layer and store the attention weights.
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights[f'decoder_layer{i + 1}_block1'] = block1
            attention_weights[f'decoder_layer{i + 1}_block2'] = block2

        return x, attention_weights
