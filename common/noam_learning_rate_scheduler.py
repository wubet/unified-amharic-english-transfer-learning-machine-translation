import tensorflow as tf


class NoamLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_factor, dmodel, warmup_steps):
        super(NoamLearningRateSchedule, self).__init__()
        self.initial_factor = initial_factor
        self.dmodel = dmodel
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return self.initial_factor * tf.math.rsqrt(self.dmodel) * tf.math.minimum(arg1, arg2)
