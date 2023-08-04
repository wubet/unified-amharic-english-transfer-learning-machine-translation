import tensorflow as tf
from transformers import TFBertModel
from models.model import Transformer
from transformers import BertConfig


# This class represents the Transformer model for training.
class CtnTransformer:
    """
    A class that represents the Custom Transformer (CTN) model.
    This class encapsulates the construction, training, and utility functions
    for a transformer model, possibly with a teacher model.
    """
    # Initialization method that takes various hyper_parameters and optional teacher model path as arguments.
    def __init__(self, input_vocab_size, target_vocab_size, d_model, num_layers, num_heads, dff, dropout_rate,
                 teacher_model_path=None):
        """
        Initializes the CtnTransformer.

        :param input_vocab_size: Size of the input vocabulary.
        :param target_vocab_size: Size of the target vocabulary.
        :param d_model: Dimension of the model.
        :param num_layers: Number of transformer layers.
        :param num_heads: Number of attention heads.
        :param dff: Dimension of the feed-forward network.
        :param dropout_rate: Dropout rate for training.
        :param teacher_model_path: (Optional) Path to the teacher model (e.g., BERT).
        """
        # Hyper_parameters and optional teacher model path are stored as instance variables.
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.teacher_model_path = teacher_model_path

        # Initialize various model-related instance variables to None.
        self.transformer_model = None
        self.train_loss = None
        self.train_accuracy = None
        self.transformer = None
        self.teacher_model = None

    # It creates a transformer model.
    def create_transformer(self):
        """
        Creates the transformer model with the given hyperparameters.
        If a teacher model path is provided, it also initializes the teacher model.
        """
        # Create input layers for the encoder and decoder.
        encoder_inputs = tf.keras.layers.Input(shape=(None,))
        decoder_inputs = tf.keras.layers.Input(shape=(None,))

        # Create the Transformer model.
        self.transformer = Transformer(self.num_layers, self.d_model, self.num_heads, self.dff,
                                       self.input_vocab_size, self.target_vocab_size, self.dropout_rate)

        # Create various masks for the input sequences.
        enc_padding_mask = self.transformer.create_padding_mask(encoder_inputs)
        look_ahead_mask = self.transformer.create_look_ahead_mask(tf.shape(decoder_inputs)[1])
        dec_padding_mask = self.transformer.create_padding_mask(encoder_inputs)

        # Training is a boolean specifying whether to apply dropout (True) or not (False).
        training = True

        # # Use the call method of the transformer to get the output.
        outputs, _, encoder_outputs = self.transformer(encoder_inputs, decoder_inputs, training, enc_padding_mask,
                                                       look_ahead_mask,
                                                       dec_padding_mask)

        # Create the complete Transformer model.
        self.transformer_model = tf.keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=encoder_outputs)

        # If a teacher model path is provided, load the teacher model (BERT).
        if self.teacher_model_path:
            # Create a BERT configuration with output_hidden_states set to True
            config = BertConfig.from_pretrained(self.teacher_model_path)
            config.output_hidden_states = True
            self.teacher_model = TFBertModel.from_pretrained(self.teacher_model_path, config=config)

    # Creates metrics for monitoring the training process.
    def create_metrics(self):
        """
        Creates metrics for monitoring the training process, including loss and accuracy.
        """
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    def get_student_encoder(self):
        """
        Constructs and returns the encoder part of the student model.

        :return: A Keras model representing the encoder.
        """
        encoder_input = tf.keras.Input(shape=(None,), dtype=tf.int32)
        enc_padding_mask = tf.keras.Input(shape=(1, 1, None), dtype=tf.float32)

        encoder_outputs = self.transformer.encoder(encoder_input, True, enc_padding_mask)
        return tf.keras.models.Model(inputs=[encoder_input, enc_padding_mask],
                                     outputs=[encoder_outputs])

    def get_student_decoder(self):
        """
        Generates a distillation rate that linearly decreases from the start_rate to end_rate
        over the course of the training epochs.

        :param current_epoch: Current epoch number (starting from 0).
        :param total_epochs: Total number of epochs for training.
        :param start_rate: Starting value for the distillation rate at epoch 0 (default 1.0).
        :param end_rate: Ending value for the distillation rate at the final epoch (default 0.0).
        :return: The calculated distillation rate for the current epoch.
        """
        # Define the layers as instance attributes in the __init__ method
        tgt_inp = tf.keras.layers.Input(shape=(None,))  # Adjust the shape as needed
        look_ahead_mask = tf.keras.layers.Input(shape=(1, None, None))
        padding_mask = tf.keras.layers.Input(shape=(1, 1, None))
        enc_output = tf.keras.layers.Input(
            shape=(None, self.d_model))  # Assuming d_model is the number of dimensions

        # Get the decoder outputs and hidden state from the decoder model
        dec_output, _ = self.transformer.decoder(
            tgt_inp, enc_output=enc_output, training=True, look_ahead_mask=look_ahead_mask,
            padding_mask=padding_mask
        )

        # Apply the final dense layer to transform to vocab size
        final_output = self.transformer.final_layer(dec_output)  # This line was missing

        return tf.keras.models.Model(
            inputs=[tgt_inp, look_ahead_mask, padding_mask, enc_output], outputs=[dec_output, final_output])

    def generate_distillation_rate(self, current_epoch, total_epochs, start_rate=1.0, end_rate=0.0):
        """
        Linearly decrease the distillation_rate from start_rate to end_rate over the total_epochs.

        Args:
            current_epoch (int): The current epoch number (starting from 0).
            total_epochs (int): The total number of epochs for training.
            start_rate (float, optional): The starting value for distillation_rate at epoch 0. Defaults to 1.0.
            end_rate (float, optional): The ending value for distillation_rate at the final epoch. Defaults to 0.0.

        Returns:
            float: The calculated distillation_rate for the current epoch.
        """
        return start_rate - (current_epoch / (total_epochs - 1)) * (start_rate - end_rate)

