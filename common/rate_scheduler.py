from abc import ABC

import tensorflow as tf


class RateScheduledOptimizers(tf.keras.optimizers.Optimizer, ABC):
    """
       Custom optimizer that wraps the Adam optimizer with a specific learning rate schedule
       meant for Neural Machine Translation (NMT) models.

       This optimizer only applies gradients to the NMT variables, adhering to the provided
       learning rate schedule.

       Attributes:
       ----------
       optimizer : tf.keras.optimizers.Adam
           The underlying Adam optimizer instance with the learning rate schedule set for NMT models.
    """
    def __init__(self, learning_rate_nmt, name="RateScheduledOptimizers", **kwargs):
        """
            Initializes the RateScheduledOptimizers.

            Parameters:
            ----------
            learning_rate_nmt : float or tf.keras.optimizers.schedules.LearningRateSchedule
                Learning rate or learning rate schedule tailored for NMT models.

            name : str, optional
                Name for the optimizer. Default is "RateScheduledOptimizers".

            **kwargs:
                Additional keyword arguments passed to the parent optimizer class.
        """
        super(RateScheduledOptimizers, self).__init__(name=name, **kwargs)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_nmt)

    def apply_gradients(self, nmt_gradients, nmt_variables):
        """
            Apply the computed gradients to the NMT variables.

            Parameters:
            ----------
            nmt_gradients : List[tf.Tensor]
                A list of gradient tensors to be applied.

            nmt_variables : List[tf.Variable]
                A list of NMT model variables to which the gradients should be applied.
                Must correspond to the gradients in the `nmt_gradients` list.
        """
        # Only apply gradients to the NMT variables
        self.optimizer.apply_gradients(zip(nmt_gradients, nmt_variables))




