from abc import ABC
# from datetime import time
# from model1 import Transformer
#
# import tensorflow as tf
#
# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
#     from_logits=True, reduction='none')
#
#
# def loss_function(real, pred):
#     mask = tf.math.logical_not(tf.math.equal(real, 0))
#     loss_ = loss_object(real, pred)
#
#     mask = tf.cast(mask, dtype=loss_.dtype)
#     loss_ *= mask
#
#     return tf.reduce_sum(loss_) / tf.reduce_sum(mask)
#
#
# def accuracy_function(real, pred):
#     accuracies = tf.equal(real, tf.argmax(pred, axis=2))
#
#     mask = tf.math.logical_not(tf.math.equal(real, 0))
#     accuracies = tf.math.logical_and(mask, accuracies)
#
#     accuracies = tf.cast(accuracies, dtype=tf.float32)
#     mask = tf.cast(mask, dtype=tf.float32)
#     return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)
#
#
# class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
#     def __init__(self, d_model, warmup_steps=4000):
#         super(CustomSchedule, self).__init__()
#
#         self.d_model = d_model
#         self.d_model = tf.cast(self.d_model, tf.float32)
#
#         self.warmup_steps = warmup_steps
#
#     def __call__(self, step):
#         arg1 = tf.math.rsqrt(step)
#         arg2 = step * (self.warmup_steps ** -1.5)
#
#         return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
#
#
# transformer = Transformer(
#     num_layers=num_layers,
#     d_model=d_model,
#     num_heads=num_heads,
#     dff=dff,
#     input_vocab_size=tokenizers.pt.get_vocab_size(),
#     target_vocab_size=tokenizers.en.get_vocab_size(),
#     pe_input=1000,
#     pe_target=1000,
#     rate=dropout_rate)
#
#
# learning_rate = CustomSchedule(d_model)
#
# optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
#                                      epsilon=1e-9)
#
# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
#
#
# def train_step(inp, tar):
#     tar_inp = tar[:, :-1]
#     tar_real = tar[:, 1:]
#
#     with tf.GradientTape() as tape:
#         predictions, _ = transformer([inp, tar_inp],
#                                      training=True)
#         loss = loss_function(tar_real, predictions)
#
#     gradients = tape.gradient(loss, transformer.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
#
#     train_loss(loss)
#     train_accuracy(accuracy_function(tar_real, predictions))
#
#
# EPOCHS = 20
#
# for epoch in range(EPOCHS):
#     start = time.time()
#
#     train_loss.reset_states()
#     train_accuracy.reset_states()
#
#     for (batch, (inp, tar)) in enumerate(train_batches):
#         train_step(inp, tar)
#
#         if batch % 50 == 0:
#             print(
#                 f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
#
#     print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')


# def train_step(self, src, tgt, optimizer, distillation_rate):
#     teacher_mask = create_padding_mask(src)
#     tgt_inp = tgt[:, :-1]
#     tgt_real = tgt[:, 1:]
#
#     nmt_mask = create_masks(tgt_inp)
#
#     with tf.GradientTape() as tape:
#         # teacher_enc_output = teacher_encoder(src, training=False, mask=teacher_mask)
#         teacher_enc_output = self.ctn_transformer.teacher_model(src, training=False)
#
#         outputs = self.ctn_transformer.teacher_model(src)
#
#         # outputs[0] contains the final activations of the model (i.e., the logits for the Masked LM).
#         # outputs[1] contains the hidden states
#         Teacher_hidden_states = outputs[1]
#
#         # student_enc_output, student_enc_hidden = self.ctn_transformer.student_encoder([src, tgt_inp], training=True)
#         # final_output, student_enc_hidden, student_enc_output = self.ctn_transformer.student_encoder([src, tgt_inp],
#         #                                                                                             training=True)
#
#         # Call the student_encoder with the appropriate inputs.
#         student_enc_output = self.ctn_transformer.student_encoder([src, tgt_inp], training=True)
#
#         # Extract the encoder output and hidden state from the student_outputs list.
#         # student_enc_output, student_enc_hidden = student_outputs[0], student_outputs[1]
#
#         # Get the final hidden state for each sequence in the batch
#         # Assuming the last axis (fully_connected_dim) represents the hidden state
#         student_enc_hidden = student_enc_output[:, -1, :]
#
#         # student_dec_output, _ = self.ctn_transformer.student_decoder(tgt_inp, student_enc_hidden,
#         #                                                              student_enc_output,
#         #                                                              training=True, mask=nmt_mask)
#         # Call the student_decoder with all required tensors
#         # student_dec_output, _ = self.ctn_transformer.student_decoder(
#         #     tgt_inp, student_enc_hidden, student_enc_output, training=True, look_ahead_mask=None,
#         #     dec_padding_mask=nmt_mask)
#
#         # Call the student_decoder with actual tensors, not the input layers
#         # student_dec_output, _ = self.ctn_transformer.student_decoder(
#         #     tgt_inp, student_enc_output, look_ahead_mask=nmt_mask, padding_mask=None, training=True
#         # )
#
#         # Get the dynamic shape of the tensor
#         dynamic_shape = tf.shape(tgt_inp)
#
#         # Extract the value of seq_len
#         seq_len = dynamic_shape[1]
#         student_dec_output = self.ctn_transformer.get_student_decoder(student_enc_output, seq_len)
#         # student_dec_output = self.ctn_transformer.student_decoder(tgt_inp, nmt_mask, None)
#         # student_dec_output = student_decoder([tgt_inp, student_enc_output, None, nmt_mask])
#
#         print("success")
#         ctnmTransformer = CTNMTransformer(Teacher_hidden_states)
#
#         # Apply dynamic switch and asymptotic distillation
#         combined_enc_output = ctnmTransformer.dynamic_switch(teacher_enc_output, student_enc_output)
#         # combined_enc_output = self.ctn_transformer.dynamic_switch(teacher_enc_output, student_enc_output,
#         # dynamic_switch_rate)
#         # combined_dec_output = self.ctn_transformer.dynamic_switch(teacher_enc_output, student_dec_output,
#         #                                                           dynamic_switch_rate)
#
#         # # Use combined_enc_output as input to the student_decoder
#         # student_dec_output, _ = self.ctn_transformer.student_decoder(tgt_inp, student_enc_hidden,
#         #                                                                                combined_enc_output,
#         #                                                                                training=True, mask=nmt_mask)
#
#         nmt_loss = self.ctn_transformer.loss_function(tgt_real, student_dec_output)
#         # nmt_loss = self.ctn_transformer.loss_function(tgt_real, combined_dec_output)
#
#         distillation_loss = mse_loss(teacher_enc_output, student_enc_output)
#
#         combined_loss = nmt_loss * distillation_rate + distillation_loss * (1 - distillation_rate)
#
#     gradients = tape.gradient(combined_loss,
#                               self.ctn_transformer.transformer_model.student_encoder.trainable_variables +
#                               self.ctn_transformer.transformer_model.student_decoder.trainable_variables)
#     optimizer.apply_gradients(
#         zip(gradients, self.ctn_transformer.transformer_model.student_encoder.trainable_variables +
#             self.ctn_transformer.transformer_model.student_decoder.trainable_variables))
#
#     return combined_loss, nmt_loss, distillation_loss
