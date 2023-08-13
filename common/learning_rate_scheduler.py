import tensorflow as tf


class CustomLRScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
       Custom Learning Rate Scheduler:
       A learning rate scheduler based on the model's dimension and warmup steps.
       This scheduler follows the learning rate strategy used in the "Attention is All You Need" paper,
       where the learning rate increases linearly for the first `warmup_steps` and then decreases proportionally
       to the inverse square root of the step number.

       Attributes:
       ----------
       d_model : int
           The dimension of the model (often referred to as depth).

       warmup_steps : int, optional (default=4000)
           Number of steps to linearly increase the learning rate before decreasing it.
    """
    def __init__(self, d_model, warmup_steps=4000):
        """
            Compute the learning rate for a given step.

            Parameters:
            ----------
            step : int
                The current training step for which the learning rate is to be computed.

            Returns:
            -------
            float
                Computed learning rate for the provided step.
        """
        super(CustomLRScheduler, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(tf.cast(step, tf.float32))
        arg2 = tf.cast(step, tf.float32) * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)



