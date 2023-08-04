from abc import ABC

import tensorflow as tf
from tensorflow.keras import layers
from encoders.encoder import Encoder
from decoders.decoder import Decoder


class Transformer(tf.keras.Model, ABC):
    """
    Complete transformer with an Encoder and a Decoder
    """
    # This method initializes an instance of the Transformer class.
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout_rate=0.1):
        """
       Initialize the Transformer class with the following parameters:

       :param num_layers: int, the number of encoder and decoder layers.
       :param d_model: int, the dimension of the key, query, and value vectors, also known as the embedding dimension.
       :param num_heads: int, the number of attention heads for the multi-head attention mechanism.
       :param dff: int, the dimensionality of the "feed forward" network inside the encoder and decoder layers.
       :param input_vocab_size: int, the size of the input vocabulary.
       :param target_vocab_size: int, the size of the target vocabulary.
       :param dropout_rate: float, optional, the dropout rate to be applied to certain layers to prevent overfitting. Default is 0.1.
       """
        # Call the parent class' constructor.
        super(Transformer, self).__init__()

        # Create an Encoder with the given parameters.
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, dropout_rate)

        # Create a Decoder with the given parameters.
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, dropout_rate)

        # Create the final layer of the model, which will transform the Decoder's output
        # into prediction scores for each possible output token.
        self.final_layer = layers.Dense(target_vocab_size)

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
             final_output: Tensor, shape (batch_size, target_seq_len, target_vocab_size), the final output of the Transformer, containing the predicted probabilities for each token in the target vocabulary.
            attention_weights: Dictionary of tensors containing all the attention weights for the decoder, each of shape (batch_size, num_heads, target_seq_len, input_seq_len).
            enc_output: Tensor, shape (batch_size, input_seq_len, fully_connected_dim), the output of the encoder.
        """

        # call self.encoder with the appropriate arguments to get the encoder output
        enc_output = self.encoder(inputs, training, enc_padding_mask)

        # call self.decoder with the appropriate arguments to get the decoder output
        dec_output, attention_weights = self.decoder(targets, enc_output, training, look_ahead_mask, dec_padding_mask)

        # pass decoder output through a linear layer and softmax (~2 lines)
        final_output = self.final_layer(dec_output)

        return final_output, attention_weights, enc_output

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
