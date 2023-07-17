import tensorflow as tf
from transformers import TFBertModel
from models.model import Transformer


# This class represents the Transformer model for training.
class CntTransformer:
    # Initialization method that takes various hyper_parameters and optional teacher model path as arguments.
    def __init__(self, input_vocab_size, target_vocab_size, d_model, num_layers, num_heads, dff, dropout_rate,
                 teacher_model_path=None):
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
        self.optimizer = None
        self.loss_object = None
        self.train_loss = None
        self.train_accuracy = None
        self.learning_rate = None
        self.transformer = None
        self.teacher_transformer = None
        self.switch_epoch = 5  # Presumably, the epoch at which some change in the training process is made.

    # It creates a transformer model.
    def create_transformer(self):
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

        # Use the call method of the transformer to get the output.
        outputs, _ = self.transformer(encoder_inputs, decoder_inputs, training, enc_padding_mask, look_ahead_mask,
                                      dec_padding_mask)

        # Create the complete Transformer model.
        self.transformer_model = tf.keras.models.Model([encoder_inputs, decoder_inputs], outputs)

        # If a teacher model path is provided, load the teacher model (BERT).
        if self.teacher_model_path:
            self.teacher_transformer = TFBertModel.from_pretrained(self.teacher_model_path)

    # Creates the optimizer for the model.
    def create_optimizer(self, learning_rate):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # Creates the loss function for the model.
    def create_cross_entropy(self):
        self.loss_object = tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)

    # Creates metrics for monitoring the training process.
    def create_metrics(self):
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    # Defines a learning rate schedule for the training process.
    def learning_rate_schedule(self, epoch):
        warmup_steps = 4000
        arg1 = tf.math.rsqrt(epoch + 1)
        arg2 = epoch * (warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    # Defines a loss function for distillation from the teacher model to the student model.
    def distillation_loss(self, labels, predictions, teacher_predictions, temperature=2.0, alpha=0.1):
        teacher_predictions = tf.stop_gradient(teacher_predictions)
        loss = self.loss_object(labels, predictions)
        teacher_loss = self.loss_object(labels, teacher_predictions)
        distillation_loss = alpha * loss + (1 - alpha) * teacher_loss
        return distillation_loss
