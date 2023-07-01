import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from transformers import TFBertModel
from model1 import Transformer


# Define the TransferLearningTransformer class
class TransferLearningTransformer:
    def __init__(self, input_vocab_size, target_vocab_size, d_model, num_layers, num_heads, dff, dropout_rate,
                 teacher_model_path=None):
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.teacher_model_path = teacher_model_path

        self.optimizer = None
        self.loss_object = None
        self.train_loss = None
        self.train_accuracy = None
        self.learning_rate = None
        self.transformer = None
        self.teacher_transformer = None
        self.switch_epoch = 5

    def create_transformer(self):
        # Create the Transformer model
        encoder_inputs = Input(shape=(None,))
        decoder_inputs = Input(shape=(None,))

        self.transformer = Transformer(self.num_layers, self.d_model, self.num_heads, self.dff,
                                       self.input_vocab_size, self.target_vocab_size, self.dropout_rate)

        enc_padding_mask = self.transformer.create_padding_mask(encoder_inputs)
        look_ahead_mask = self.transformer.create_look_ahead_mask(tf.shape(decoder_inputs)[1])
        dec_padding_mask = self.transformer.create_padding_mask(encoder_inputs)

        encoder_outputs = self.transformer.encoder(encoder_inputs, enc_padding_mask)
        decoder_outputs, _ = self.transformer.decoder(decoder_inputs, encoder_outputs, look_ahead_mask,
                                                      dec_padding_mask)

        outputs = self.transformer.final_layer(decoder_outputs)

        self.transformer_model = Model([encoder_inputs, decoder_inputs], outputs)

        if self.teacher_model_path:
            # Load teacher model (BERT)
            self.teacher_transformer = TFBertModel.from_pretrained(self.teacher_model_path)

    def create_optimizer(self, learning_rate):
        # Create optimizer and loss function
        self.optimizer = Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.loss_object = SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def create_metrics(self):
        # Create metrics for monitoring training
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = SparseCategoricalAccuracy(name='train_accuracy')

    def learning_rate_schedule(self, epoch):
        # Learning rate schedule
        warmup_steps = 4000
        arg1 = tf.math.rsqrt(epoch + 1)
        arg2 = epoch * (warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def distillation_loss(self, labels, predictions, teacher_predictions, temperature=2.0, alpha=0.1):

        # Asymptotic distillation loss with teacher and student predictions
        teacher_predictions = tf.stop_gradient(teacher_predictions)

        loss = self.loss_object(labels, predictions)
        teacher_loss = self.loss_object(labels, teacher_predictions)

        distillation_loss = alpha * loss + (1 - alpha) * teacher_loss

        return distillation_loss

    # def distillation_loss(self, labels, predictions, teacher_predictions, temperature=2.0, alpha=0.1):
    #     # Asymptotic distillation loss with teacher and student predictions
    #     teacher_predictions = tf.stop_gradient(teacher_predictions)
    #
    #     # Compute the cross-entropy loss between student predictions and labels
    #     student_loss = self.loss_object(labels, predictions)
    #
    #     # Compute the cross-entropy loss between teacher predictions and labels
    #     teacher_loss = self.loss_object(labels, teacher_predictions)
    #
    #     # Apply temperature scaling to both student and teacher predictions
    #     scaled_student_predictions = predictions / temperature
    #     scaled_teacher_predictions = teacher_predictions / temperature
    #
    #     # Compute the softmax cross-entropy loss between the scaled student and teacher predictions
    #     kd_loss = self.loss_object(scaled_teacher_predictions, scaled_student_predictions)
    #
    #     # Compute the final distillation loss as a weighted sum of the student loss and the knowledge distillation loss
    #     distillation_loss = alpha * student_loss + (1 - alpha) * kd_loss
    #
    #     return distillation_loss

