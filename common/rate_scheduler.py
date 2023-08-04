from abc import ABC

import tensorflow as tf


class RateScheduledOptimizers(tf.keras.optimizers.Optimizer, ABC):
    def __init__(self, learning_rate_nmt, name="RateScheduledOptimizers", **kwargs):
        super(RateScheduledOptimizers, self).__init__(name=name, **kwargs)
        self.optimizer_nmt = tf.keras.optimizers.Adam(learning_rate=learning_rate_nmt)

    def apply_gradients(self, nmt_gradients, nmt_variables):
        # Only apply gradients to the NMT variables
        self.optimizer_nmt.apply_gradients(zip(nmt_gradients, nmt_variables))

    # You may also need to define other required methods here


