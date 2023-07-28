import tensorflow as tf


class RateScheduledOptimizers:
    def __init__(self, learning_rate_bert, learning_rate_nmt):
        self.optimizer_bert = tf.keras.optimizers.Adam(learning_rate=learning_rate_bert)
        self.optimizer_nmt = tf.keras.optimizers.Adam(learning_rate=learning_rate_nmt)

    def apply_gradients(self, bert_gradients, nmt_gradients, bert_variables, nmt_variables):
        self.optimizer_bert.apply_gradients(zip(bert_gradients, bert_variables))
        self.optimizer_nmt.apply_gradients(zip(nmt_gradients, nmt_variables))
