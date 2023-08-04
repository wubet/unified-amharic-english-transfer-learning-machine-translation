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
# class CtnTransformer:
#     # Initialization method that takes various hyper_parameters and optional teacher model path as arguments.
#     def __init__(self, input_vocab_size, target_vocab_size, d_model, num_layers, num_heads, dff, dropout_rate,
#                  teacher_model_path=None):
#         # Hyper_parameters and optional teacher model path are stored as instance variables.
#         self.input_vocab_size = input_vocab_size
#         self.target_vocab_size = target_vocab_size
#         self.d_model = d_model
#         self.num_layers = num_layers
#         self.num_heads = num_heads
#         self.dff = dff
#         self.dropout_rate = dropout_rate
#         self.teacher_model_path = teacher_model_path
#
#         # Initialize various model-related instance variables to None.
#         self.transformer_model = None
#         self.optimizer = None
#         # self.loss_object = None
#         self.train_loss = None
#         self.train_accuracy = None
#         self.learning_rate = None
#         self.transformer = None
#         self.teacher_model = None
#         self.student_encoder = None
#         self.student_decoder = None
#         self.encoder_input_layer = None
#         self.enc_padding_mask_layer = None
#         self.switch_epoch = 5  # Presumably, the epoch at which some change in the training process is made.
#
#     # It creates a transformer model.
#     def create_transformer(self):
#         # Create input layers for the encoder and decoder.
#         encoder_inputs = tf.keras.layers.Input(shape=(None,))
#         decoder_inputs = tf.keras.layers.Input(shape=(None,))
#         # Create input layers for the encoder and decoder.
#         # encoder_inputs = tf.keras.layers.Input(shape=(None, self.input_vocab_size))
#         # decoder_inputs = tf.keras.layers.Input(shape=(None, self.target_vocab_size))
#
#         # Create the Transformer model.
#         self.transformer = Transformer(self.num_layers, self.d_model, self.num_heads, self.dff,
#                                        self.input_vocab_size, self.target_vocab_size, self.dropout_rate)
#
#         # Create various masks for the input sequences.
#         enc_padding_mask = self.transformer.create_padding_mask(encoder_inputs)
#         look_ahead_mask = self.transformer.create_look_ahead_mask(tf.shape(decoder_inputs)[1])
#         dec_padding_mask = self.transformer.create_padding_mask(encoder_inputs)
#
#         # Training is a boolean specifying whether to apply dropout (True) or not (False).
#         training = True
#
#         # # Use the call method of the transformer to get the output.
#         outputs, _, encoder_outputs = self.transformer(encoder_inputs, decoder_inputs, training, enc_padding_mask,
#                                                        look_ahead_mask,
#                                                        dec_padding_mask)
#
#         # encoder_outputs, encoder_hidden = self.transformer.encoder(encoder_inputs, training, enc_padding_mask)
#
#         # Now, create a separate layer for encoder_outputs and connect it to the model output.
#         # encoder_outputs_layer = tf.keras.layers.Lambda(lambda x: x)  # Identity layer for encoder_outputs
#         # encoder_outputs = encoder_outputs_layer(encoder_outputs)
#
#         # Create the self.student_encoder model directly.
#         # self.student_encoder = tf.keras.models.Model([encoder_inputs, decoder_inputs], encoder_outputs)
#
#         # self.student_encoder = self.transformer.encoder()
#         # tf.keras.models.Model([encoder_inputs, enc_padding_mask], encoder_outputs)
#
#         # Model definition (or wherever you define your layers)
#         # self.encoder_input_layer = tf.keras.Input(shape=(None,), dtype=tf.int32)
#         # self.student_encoder = tf.keras.models.Model([encoder_inputs], encoder_outputs)
#
#         # Create the complete Transformer model.
#         self.enc_padding_mask_layer = tf.keras.Input(shape=(1, 1, None), dtype=tf.float32)
#
#         # self.student_decoder = self.get_student_decoder()
#
#         # Create the complete Transformer model.
#         self.transformer_model = tf.keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=encoder_outputs)
#
#         # If a teacher model path is provided, load the teacher model (BERT).
#         if self.teacher_model_path:
#             # Create a BERT configuration with output_hidden_states set to True
#             config = BertConfig.from_pretrained(self.teacher_model_path)
#             config.output_hidden_states = True
#             self.teacher_model = TFBertModel.from_pretrained(self.teacher_model_path, config=config)
#
#     # # Creates the optimizer for the model.
#     # def create_optimizer(self, learning_rate):
#     #     self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
#     #
#     # # Creates the loss function for the model.
#     # def create_cross_entropy(self):
#     #     self.loss_object = tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)
#
#     # Creates metrics for monitoring the training process.
#     def create_metrics(self):
#         self.train_loss = tf.keras.metrics.Mean(name='train_loss')
#         self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
#
#     # Defines a learning rate schedule for the training process.
#     def learning_rate_schedule(self, epoch):
#         warmup_steps = 4000
#         arg1 = tf.math.rsqrt(epoch + 1)
#         arg2 = epoch * (warmup_steps ** -1.5)
#         return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
#
#     # Defines a loss function for distillation from the teacher model to the student model.
#     def distillation_loss(self, labels, student_predictions, teacher_predictions, temperature=2.0, alpha=0.1):
#         # teacher_predictions = tf.stop_gradient(teacher_predictions)
#         # loss = self.loss_object(labels, student_predictions)
#         # teacher_loss = self.loss_object(labels, teacher_predictions)
#         # distillation_loss = alpha * loss + (1 - alpha) * teacher_loss
#         # Extract the relevant hidden states from the teacher predictions
#         teacher_hidden_states = teacher_predictions.last_hidden_state  # Assuming 'last_hidden_state' contains the hidden states
#         print("teacher_hidden_states: ", teacher_hidden_states.shape)
#
#         # Stop gradient to prevent backpropagation to the teacher model
#         teacher_hidden_states = tf.stop_gradient(teacher_hidden_states)
#
#         # Compute the distillation loss using mean squared error (MSE)
#         distillation_loss = tf.reduce_mean(tf.square(teacher_hidden_states - student_predictions))
#         return distillation_loss
#
#     # Define the distillation loss function
#     def asymptotic_distillation_loss(self, labels, student_predictions, teacher_predictions):
#         teacher_predictions = tf.stop_gradient(teacher_predictions)
#         loss = tf.keras.losses.sparse_categorical_crossentropy(labels, student_predictions, from_logits=True)
#         teacher_loss = tf.keras.losses.sparse_categorical_crossentropy(labels, teacher_predictions, from_logits=True)
#         distillation_loss = tf.reduce_mean(loss) + tf.reduce_mean(teacher_loss)
#         return distillation_loss
#
#     # Define the cross-entropy loss function
#     def cross_entropy_loss(self, labels, predictions):
#         loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions, from_logits=True)
#         return tf.reduce_mean(loss)
#
#     def adjust_learning_rate(self, epoch):
#         # Adjust the learning rate of the optimizer based on the epoch and learning rate schedule
#         new_learning_rate = self.learning_rate_schedule(epoch)
#         self.optimizer.learning_rate.assign(new_learning_rate)
#
#     def dynamic_switching_gate(self, pretrained_hidden_state, nmt_hidden_state):
#         gate = tf.keras.activations.sigmoid(
#             tf.keras.layers.Dense(1)(tf.concat([pretrained_hidden_state, nmt_hidden_state], axis=-1)))
#         fused_representation = gate * pretrained_hidden_state + (1 - gate) * nmt_hidden_state
#         return fused_representation
#
#     def dynamic_switching_gate(self, source_language_input_ids, h_lm, h_nmt):
#         # Compute the context gate
#         gate_inputs = tf.concat([h_lm, h_nmt], axis=-1)
#         gate_weights = tf.keras.layers.Dense(1, activation='sigmoid')(gate_inputs)
#         context_gate = tf.multiply(h_lm, gate_weights) + tf.multiply(h_nmt, 1 - gate_weights)
#
#         # Fuse the context gate with the source input embeddings
#         fused_representation = tf.multiply(source_language_input_ids, context_gate)
#
#         return fused_representation
#
#     def validate_teacher_predictions(self, teacher_predictions, student_predictions):
#         # Check if the teacher_predictions variable contains the expected key
#         if "last_hidden_state" not in teacher_predictions:
#             raise ValueError("Teacher predictions do not contain the 'last_hidden_state' key.")
#
#         # Extract the hidden states from the teacher predictions
#         teacher_hidden_states = teacher_predictions["last_hidden_state"]
#
#         # Get the shapes of teacher_hidden_states and student_predictions
#         teacher_shape = tf.shape(teacher_hidden_states)
#         student_shape = tf.shape(student_predictions)
#
#         # Check if the shapes match
#         shape_equal = tf.reduce_all(tf.equal(teacher_shape, student_shape))
#         if not shape_equal:
#             raise ValueError("Shapes of teacher_hidden_states and student_predictions do not match. "
#                              f"Teacher shape: {teacher_shape}, Student shape: {student_shape}")
#
#         # Check if the hidden size (dimension) matches
#         hidden_size_teacher = teacher_shape[-1]
#         hidden_size_student = student_shape[-1]
#         if hidden_size_teacher != hidden_size_student:
#             raise ValueError("Hidden sizes of teacher_hidden_states and student_predictions do not match. "
#                              f"Teacher hidden size: {hidden_size_teacher}, Student hidden size: {hidden_size_student}")
#
#     def dynamic_switch(self, teacher_enc_output, student_enc_output, dynamic_switch_rate):
#         switch_rate = tf.sigmoid(dynamic_switch_rate)
#         return switch_rate * teacher_enc_output + (1 - switch_rate) * student_enc_output
#
#         # # Dynamic switch
#         # context_gate = tf.keras.layers.Dense(1, activation='sigmoid')
#         # g = context_gate(tf.concat([teacher_enc_output, student_enc_output], axis=-1))
#         # switched_output = g * teacher_enc_output + (1 - g) * student_enc_output
#         # return switched_output
#
#     # def get_student_encoder(self, encoder_inputs, enc_padding_mask):
#     #
#     #     # Then, we'll apply this Encoder to the inputs.
#     #     encoder_outputs = self.transformer.encoder(encoder_inputs, True, enc_padding_mask)
#     #
#     #     # Finally, we'll create and return a Model.
#     #     return tf.keras.models.Model([encoder_inputs, enc_padding_mask], encoder_outputs)
#
#     def student_encoder(self, encoder_inputs, training, enc_padding_mask):
#         # encoder_inputs, decoder_inputs = inputs
#         # encoder_padding_mask = self.create_padding_mask(encoder_inputs)
#         # decoder_padding_mask = self.create_padding_mask(decoder_inputs)
#         # look_ahead_mask = self.create_look_ahead_mask(tf.shape(decoder_inputs)[1])
#
#         encoder_outputs = self.transformer.encoder(encoder_inputs, training, enc_padding_mask)
#
#         # decoder_outputs, decoder_hidden = self.transformer.decoder(
#         #     decoder_inputs, encoder_outputs, training, look_ahead_mask, dec_padding_mask
#         # )
#
#         return [encoder_outputs]
#
#     def get_student_encoder(self):
#         encoder_input = tf.keras.Input(shape=(None,), dtype=tf.int32)
#         enc_padding_mask = tf.keras.Input(shape=(1, 1, None), dtype=tf.float32)
#
#         encoder_outputs = self.transformer.encoder(encoder_input, True, enc_padding_mask)
#         return tf.keras.models.Model(inputs=[encoder_input, enc_padding_mask],
#                                      outputs=[encoder_outputs])
#
#     # def student_encoder(self, inputs, training, enc_padding_mask, dec_padding_mask, look_ahead_mask):
#     #     encoder_inputs, decoder_inputs = inputs
#     #     # encoder_padding_mask = self.create_padding_mask(encoder_inputs)
#     #     # decoder_padding_mask = self.create_padding_mask(decoder_inputs)
#     #     # look_ahead_mask = self.create_look_ahead_mask(tf.shape(decoder_inputs)[1])
#     #
#     #     encoder_outputs, encoder_hidden = self.transformer.encoder(encoder_inputs, training, enc_padding_mask)
#     #
#     #     decoder_outputs, decoder_hidden = self.transformer.decoder(
#     #         decoder_inputs, encoder_outputs, training, look_ahead_mask, dec_padding_mask
#     #     )
#     #
#     #     return [encoder_outputs, encoder_hidden]
#
#     # def get_student_decoder(self, enc_output, target_seq_len):
#     #     # tar = tf.keras.layers.Input(shape=(None,))
#     #     # look_ahead_mask = tf.keras.layers.Input(shape=(1, None, None))
#     #     # padding_mask = tf.keras.layers.Input(shape=(1, 1, None))
#     #     # dec_output, attention_weights = self.transformer.decoder(x=tar, enc_output=enc_output, training=True,
#     #     #                                                          look_ahead_mask=look_ahead_mask,
#     #     #                                                          padding_mask=padding_mask)
#     #     #
#     #     # return tf.keras.models.Model(inputs=[tar, enc_output, look_ahead_mask, padding_mask],
#     #     #                              outputs=[dec_output, attention_weights])
#     #     # tar = tf.keras.layers.Input(shape=(None,))
#     #     # look_ahead_mask = tf.keras.layers.Input(shape=(1, None, None))
#     #     # padding_mask = tf.keras.layers.Input(shape=(1, 1, None))
#     #     # dec_output, _ = self.transformer.decoder(x=tar, enc_output=enc_output, training=True,
#     #     #                                          look_ahead_mask=look_ahead_mask,
#     #     #                                          padding_mask=padding_mask)
#     #     #
#     #     # return tf.keras.models.Model(inputs=[tar, look_ahead_mask, padding_mask],
#     #     #                              outputs=dec_output)
#     #     # Create input layers for the decoder.
#     #     # Create input layers for the decoder.
#     #     tar = tf.keras.layers.Input(shape=(target_seq_len,))
#     #     look_ahead_mask = tf.keras.layers.Input(shape=(1, None, None))
#     #     padding_mask = tf.keras.layers.Input(shape=(1, 1, None))
#     #
#     #     # Get the decoder outputs and hidden state from the decoder model
#     #     decoder_inputs = [tar, look_ahead_mask, padding_mask]  # Remove enc_output from decoder inputs
#     #     dec_output, _ = self.transformer.decoder(
#     #         tar, enc_output=enc_output, training=True, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask
#     #     )
#     #
#     #     return tf.keras.models.Model(inputs=decoder_inputs, outputs=dec_output)
#
#     def get_student_decoder(self):
#         # Define the layers as instance attributes in the __init__ method
#         tgt_inp = tf.keras.layers.Input(shape=(None,))  # Adjust the shape as needed
#         look_ahead_mask = tf.keras.layers.Input(shape=(1, None, None))
#         padding_mask = tf.keras.layers.Input(shape=(1, 1, None))
#         enc_output = tf.keras.layers.Input(
#             shape=(None, self.d_model))  # Assuming d_model is the number of dimensions
#
#         # Get the decoder outputs and hidden state from the decoder model
#         dec_output, _ = self.transformer.decoder(
#             tgt_inp, enc_output=enc_output, training=True, look_ahead_mask=look_ahead_mask,
#             padding_mask=padding_mask
#         )
#
#         # Apply the final dense layer to transform to vocab size
#         final_output = self.transformer.final_layer(dec_output)  # This line was missing
#
#         return tf.keras.models.Model(
#             inputs=[tgt_inp, look_ahead_mask, padding_mask, enc_output], outputs=[dec_output, final_output])
#
#     # def loss_function(self, real, pried):
#     #     loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
#     #     mask = tf.math.logical_not(tf.math.equal(real, 0))
#     #     loss_ = loss_object(real, pried)
#     #
#     #     mask = tf.cast(mask, dtype=loss_.dtype)
#     #     loss_ *= mask
#     #     return tf.reduce_sum(loss_) / tf.reduce_sum(mask)
#
#     def loss_function(self, real, pried):
#         loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
#         mask = tf.math.logical_not(tf.math.equal(real, 0))
#         loss_ = loss_object(real, pried)
#
#         mask = tf.cast(mask, dtype=loss_.dtype)
#         loss_ *= mask
#         return tf.reduce_sum(loss_) / tf.reduce_sum(mask)
#
#     def generate_distillation_rate(self, current_epoch, total_epochs, start_rate=1.0, end_rate=0.0):
#         """
#         Linearly decrease the distillation_rate from start_rate to end_rate over the total_epochs.
#
#         Args:
#             current_epoch (int): The current epoch number (starting from 0).
#             total_epochs (int): The total number of epochs for training.
#             start_rate (float, optional): The starting value for distillation_rate at epoch 0. Defaults to 1.0.
#             end_rate (float, optional): The ending value for distillation_rate at the final epoch. Defaults to 0.0.
#
#         Returns:
#             float: The calculated distillation_rate for the current epoch.
#         """
#         return start_rate - (current_epoch / (total_epochs - 1)) * (start_rate - end_rate)
#
#     # def create_padding_mask(self, seq):
#     #     seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
#     #     return seq[:, tf.newaxis, tf.newaxis, :]  # add extra dimensions to match the BERT model's expected input
#     #
#     #

