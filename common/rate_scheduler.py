import tensorflow as tf


class RateScheduledOptimizers:
    def __init__(self, learning_rate_nmt):
        # Only create an optimizer for the NMT model
        self.optimizer_nmt = tf.keras.optimizers.Adam(learning_rate=learning_rate_nmt)

    def apply_gradients(self, nmt_gradients, nmt_variables):
        # Only apply gradients to the NMT variables
        self.optimizer_nmt.apply_gradients(zip(nmt_gradients, nmt_variables))

