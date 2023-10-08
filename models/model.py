from abc import ABC

import tensorflow as tf
from encoders.encoder import Encoder
from decoders.decoder import Decoder
from keras import layers


class Transformer(tf.keras.Model, ABC):
    """
    Complete transformer with an Encoder and a Decoder
    """
    # This method initializes an instance of the Transformer class.
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size,
                 position_encoding_input, position_encoding_target, dropout_rate=0.1):
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
        self.encoder = Encoder(
            num_layers, d_model, num_heads, dff, input_vocab_size,
            position_encoding_input, dropout_rate
        )

        # Create a Decoder with the given parameters.
        self.decoder = Decoder(
            num_layers, d_model, num_heads, dff, target_vocab_size,
            position_encoding_target, dropout_rate
        )

        # Create the final layer of the model, which will transform the Decoder's output
        # into prediction scores for each possible output token.
        self.final_layer = layers.Dense(target_vocab_size)

    def call(self, inputs, targets=None, training=False, enc_padding_mask=None, look_ahead_mask=None,
             dec_padding_mask=None, external_enc_output=None):
        """
          Parameters:
          ----------
          inputs : tf.Tensor
              Tensor containing source sequences for the encoder.

          targets : tf.Tensor
              Tensor containing target sequences for the decoder.

          training : bool
              Indicates whether the model is in training mode or inference mode.

          enc_padding_mask : tf.Tensor
              Padding mask for the encoder to ignore certain words in the inputs. This is typically used
              to prevent attention mechanism from attending to padding tokens.

          look_ahead_mask : tf.Tensor
              The look-ahead mask to prevent the decoder from attending to future tokens in the sequence,
              used in the self-attention mechanism.

          dec_padding_mask : tf.Tensor
              Padding mask for the decoder to ignore certain words in the decoded sequence.

          external_enc_output : tf.Tensor, optional (default=None)
              If provided, this tensor is used as the encoder's output, bypassing the internal encoder.
              This is useful for scenarios like knowledge distillation where the encoder output from a
              teacher model might be provided directly.

          Returns:
          -------
          final_output : tf.Tensor
              The final output of the decoder.

          attention_weights : dict
              Dictionary containing attention weights from different layers.
        """
        # If external encoder output is provided, use it. Otherwise, compute it.
        enc_output = external_enc_output if external_enc_output is not None else self.encoder(inputs, training,
                                                                                              enc_padding_mask)

        # call self.decoder with the appropriate arguments to get the decoder output
        dec_output, attention_weights = self.decoder(targets, enc_output, training, look_ahead_mask, dec_padding_mask)

        # pass decoder output through a linear layer and softmax
        final_output = self.final_layer(dec_output)

        return final_output, attention_weights