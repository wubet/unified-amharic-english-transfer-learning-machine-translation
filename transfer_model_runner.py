import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy


# class TrainTransferTransformer:
#     def __init__(self, transformer, learning_rate, checkpoint_path):
#         self.transformer = transformer
#         self.learning_rate = learning_rate
#         self.checkpoint_path = checkpoint_path
#         self.optimizer = Adam(learning_rate=self.learning_rate)
#         self.loss_object = SparseCategoricalCrossentropy(from_logits=True)
#         self.train_loss = tf.keras.metrics.Mean(name='train_loss')
#         self.train_accuracy = SparseCategoricalAccuracy(name='train_accuracy')
#         self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.transformer)
#
#     @tf.function
#     def train_step(self, inp, tar):
#         tar_inp = tar[:, :-1]
#         tar_real = tar[:, 1:]
#
#         enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(inp, tar_inp)
#
#         with tf.GradientTape() as tape:
#             predictions, _ = self.transformer(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
#             loss = self.loss_function(tar_real, predictions)
#
#         gradients = tape.gradient(loss, self.transformer.trainable_variables)
#         self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))
#
#         self.train_loss(loss)
#         self.train_accuracy(tar_real, predictions)
#
#     def create_masks(self, inp, tar):
#         enc_padding_mask = self.transformer.create_padding_mask(inp)
#         dec_padding_mask = self.transformer.create_padding_mask(inp)
#
#         look_ahead_mask = self.transformer.create_look_ahead_mask(tf.shape(tar)[1])
#         dec_target_padding_mask = self.transformer.create_padding_mask(tar)
#         combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
#
#         return enc_padding_mask, combined_mask, dec_padding_mask
#
#     def loss_function(self, real, pred):
#         mask = tf.math.logical_not(tf.math.equal(real, 0))
#         loss = self.loss_object(real, pred)
#
#         mask = tf.cast(mask, dtype=loss.dtype)
#         loss *= mask
#
#         return tf.reduce_mean(loss)
#
#     def train(self, train_dataset, epochs):
#         self.train_loss.reset_states()
#         self.train_accuracy.reset_states()
#
#         ckpt_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_path, max_to_keep=5)
#
#         for epoch in range(epochs):
#             for (batch, (inp, tar)) in enumerate(train_dataset):
#                 self.train_step(inp, tar)
#
#                 if batch % 50 == 0:
#                     print(f'Epoch {epoch + 1} Batch {batch} Loss {self.train_loss.result():.4f} Accuracy {self.train_accuracy.result():.4f}')
#
#             if (epoch + 1) % 5 == 0:
#                 ckpt_save_path = ckpt_manager.save()
#                 print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')
#
#             print(f'Epoch {epoch + 1} Loss {self.train_loss.result():.4f} Accuracy {self.train_accuracy.result():.4f}')
#
#         return self.checkpoint_path


class TrainTransferTransformer:
    def __init__(self, transformer, optimizer, loss_object, train_loss, train_accuracy, checkpoint_path):
        self.transformer = transformer
        self.optimizer = optimizer
        self.loss_object = loss_object
        self.train_loss = train_loss
        self.train_accuracy = train_accuracy
        # "./checkpoints/ckpt"
        self.checkpoint_path = checkpoint_path

    def create_masks(self, inp, tar):
        encoder_padding_mask = self.transformer.create_padding_mask(inp)
        decoder_padding_mask = self.transformer.create_padding_mask(inp)
        look_ahead_mask = self.transformer.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.transformer.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        return encoder_padding_mask, combined_mask, decoder_padding_mask

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask

        return tf.reduce_mean(loss)

    def train_step(self, inp, tar, tchr=None, use_distillation=False):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        encoder_padding_mask, combined_mask, decoder_padding_mask = self.create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = self.transformer(
                inp, tar_inp, True, encoder_padding_mask, combined_mask, decoder_padding_mask
            )

            if use_distillation and tchr is not None:
                tchr_predictions, _ = tchr.transformer(
                    inp, tar_inp, True, encoder_padding_mask, combined_mask, decoder_padding_mask
                )
                distillation_loss = self.loss_object(tar_real, tchr_predictions)
                loss = distillation_loss + self.loss_object(tar_real, predictions)
            else:
                loss = self.loss_object(tar_real, predictions)

        gradients = tape.gradient(loss, self.transformer.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(tar_real, predictions)

    def train(self, train_dataset, teacher=None, use_distillation=False, epochs=20, rate_schedule=False):
        for epoch in range(epochs):
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            if rate_schedule:
                self.optimizer.lr.assign(self.rate_scheduler(epoch))

            for (batch, (inp, tar)) in enumerate(train_dataset):
                self.train_step(inp, tar, teacher, use_distillation)

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = self.checkpoint_path
                self.transformer.save_weights(ckpt_save_path)
                print(f"Saving checkpoint for epoch {epoch + 1}")
                print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')

            print(
                f"Epoch {epoch + 1} Loss {self.train_loss.result():.4f} Accuracy {self.train_accuracy.result():.4f}"
            )

    def rate_scheduler(self, epoch, warmup_steps=4000):
        arg1 = tf.math.rsqrt(tf.cast(epoch + 1, tf.float32))
        arg2 = epoch * (warmup_steps ** -1.5)
        return tf.math.rsqrt(tf.cast(self.transformer.d_model, tf.float32)) * tf.math.minimum(arg1, arg2)


class TrainTransferLearningTransformer:
    def __init__(self, transfer_learning_transformer_model):
        self.transfer_learning_transformer_model = transfer_learning_transformer_model

    def train_step(self, source_language, target_language):
        tar_input = target_language[:, :-1]
        tar_real = target_language[:, 1:]

        with tf.GradientTape() as tape:
            predictions, _, *extra = self.transfer_learning_transformer_model.transformer_model([source_language, tar_input], training=True)
            # if self.transfer_learning_transformer_model.teacher_model_path and self.transfer_learning_transformer_model.teacher_transformer:
            #     print(source_language.shape)
            #     print(tar_input.shape)
            #     teacher_predictions = self.transfer_learning_transformer_model.teacher_transformer([source_language, tar_input],
            #                                                                                        training=False)[0]
            #     loss = self.transfer_learning_transformer_model.distillation_loss(tar_real, predictions, teacher_predictions)
            # else:
            #     loss = self.transfer_learning_transformer_model.loss_object(tar_real, predictions)
            loss = self.transfer_learning_transformer_model.loss_object(tar_real, predictions)
        gradients = tape.gradient(loss, self.transfer_learning_transformer_model.transfer_learning_transformer_model.trainable_variables)
        self.transfer_learning_transformer_model.optimizer.apply_gradients(
            zip(gradients, self.transfer_learning_transformer_model.transfer_learning_transformer_model.trainable_variables))

        self.transfer_learning_transformer_model.train_loss(loss)
        self.transfer_learning_transformer_model.train_accuracy(tar_real, predictions)

    def train(self, dataset, epochs):
        for epoch in range(epochs):
            self.transfer_learning_transformer_model.train_loss.reset_states()
            self.transfer_learning_transformer_model.train_accuracy.reset_states()

            # for (batch, (source_language, target_language)) in enumerate(dataset):
            #     print(f"Type of target_language: {type(target_language)}")
            #     print(f"Shape of target_language: {tf.shape(target_language)}")
            #     self.train_step(source_language, target_language)
            # for i, data in enumerate(dataset.take(5)):
            #     print(f"Element {i}: {data}")

            for batch in dataset:
                (source_language_input_ids, source_language_attention_mask), (
                    target_language_input_ids, target_language_attention_mask) = batch
                self.train_step(source_language_input_ids, target_language_input_ids)

            print(
                f'Epoch {epoch + 1} Loss {self.transfer_learning_transformer_model.train_loss.result():.4f} Accuracy {self.transfer_learning_transformer_model.train_accuracy.result():.4f}')
