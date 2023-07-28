import tensorflow as tf
from tensorflow.keras.models import Model
from transformers import TFBertModel, BertModel, BertTokenizer


class DistillationBertNMT(Model):
    def __init__(self, student, teacher):
        super(DistillationBertNMT, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(self, optimizer, metrics, distillation_loss_fn, alpha=0.1):
        super(DistillationBertNMT, self).compile(optimizer=optimizer, metrics=metrics)
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha

    def train_step(self, data):
        x, y = data
        teacher_predictions = self.teacher(x, training=False)
        with tf.GradientTape() as tape:
            student_predictions = self.student(x, training=True)
            loss = self.distillation_loss_fn(y, student_predictions)
            loss += self.alpha * self.distillation_loss_fn(teacher_predictions, student_predictions)

        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, student_predictions)
        return {m.name: m.result() for m in self.metrics}


class DistillBERTNMT(tf.keras.Model):
    def __init__(self, teacher_model, student_model, temperature=2.0):
        super(DistillBERTNMT, self).__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature

    def call(self, inputs):
        teacher_logits = self.teacher_model(inputs)
        student_logits = self.student_model(inputs)

        loss = tf.keras.losses.KLDivergence(
            reduction=tf.keras.losses.Reduction.AUTO,
            name='kl_divergence'
        )
        distillation_loss = loss(
            teacher_logits / self.temperature,
            student_logits / self.temperature
        )
        return distillation_loss

bert_model = BertModel.from_pretrained('bert-base-uncased')
# Define or load your seq2seq model here
# seq2seq_model
#
# model = DistillBERTNMT(bert_model, seq2seq_model)