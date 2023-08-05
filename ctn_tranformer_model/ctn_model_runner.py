import tensorflow as tf
from common.utils import *
from common.visualization import *
from ctn_tranformer_model.ctnmt import CTNMTransformer
from common.rate_scheduler import RateScheduledOptimizers
import pandas as pd
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# The class for training the transformer model.
class TrainCntModel:
    # Constructor to initialize the training model
    def __init__(self, ctn_transformer, check_point_path, learning_rate, source_language,
                 target_language, padding_value=0):
        # Assigning the transformer model to be trained
        self.ctn_transformer = ctn_transformer
        # Path for checkpoint saving
        self.check_point_path = check_point_path
        # Learning rate for the training process
        self.learning_rate = learning_rate
        # Value for padding sequences
        self.padding_value = padding_value
        # Source language ID
        self.source_language = source_language
        # Target language ID
        self.target_language = target_language
        # Initializing checkpoint for saving the model and optimizer
        # Define a loss object
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=RateScheduledOptimizers(
            learning_rate_nmt=self.learning_rate), net=self.ctn_transformer.transformer)
        # Checkpoint manager to handle checkpoint saving
        self.manager = tf.train.CheckpointManager(self.ckpt, './' + check_point_path, max_to_keep=3)
        # Path for saving visualization data as CSV
        self.csv_path = 'outputs/visualization_data.csv'
        # If a checkpoint exists, restore from it and load metrics
        if self.manager.latest_checkpoint:
            self.ckpt.restore(self.manager.latest_checkpoint)
            print("Restored from checkpoint:", self.manager.latest_checkpoint)
            # Load previous accuracy, loss, learning rates, and steps from CSV file
            if os.path.exists(self.csv_path):
                df = pd.read_csv(self.csv_path)
                self.accuracies = df['accuracy'].tolist()
                self.losses = df['loss'].tolist()
                self.learning_rates = df['learning_rate'].tolist()
                self.steps = df['step'].tolist()
            else:
                self.accuracies = []
                self.losses = []
                self.learning_rates = []
                self.steps = []
        else:
            # If no checkpoints found, initialize metrics as empty lists
            print("No checkpoint found. Starting a new training.")
            self.accuracies = []
            self.losses = []
            self.learning_rates = []
            self.steps = []

    # @tf.function
    def train_step(self, src, tgt, optimizer, distillation_rate):
        # Prepare target input and real target by shifting tokens
        tgt_inp = tgt[:, :-1]
        tgt_real = tgt[:, 1:]
        # Create padding mask for encoder
        enc_padding_mask = create_padding_mask(src)
        # Determine target sequence length
        tgt_shape = tf.shape(tgt_inp)
        tgt_seq_len = tgt_shape[1]
        # Create look-ahead mask for target sequence
        look_ahead_mask = create_look_ahead_attention_mask(tgt_seq_len)
        # Create padding mask for NMT
        nmt_mask = create_padding_mask(src)
        # Initialize variable to store teacher hidden encoder output
        teacher_hidden_enc = None
        # Record operations for backpropagation
        with tf.GradientTape() as tape:
            # Get teacher encoder output for source input
            teacher_enc_output = self.ctn_transformer.teacher_model(src, training=False)
            # Extract last hidden state from teacher encoder
            if 'last_hidden_state' in teacher_enc_output:
                teacher_hidden_enc = teacher_enc_output['last_hidden_state']
            # Determine weight size from teacher hidden encoder shape
            teacher_hidden_shape = tf.shape(teacher_hidden_enc)
            weight_size = teacher_hidden_shape[2]
            # Get student encoder output for source input
            student_enc_output = self.ctn_transformer.get_student_encoder()([src, enc_padding_mask], training=True)
            # Create a CTNMTransformer object with the weight size
            ctnm_transformer = CTNMTransformer(weight_size)
            # Combine teacher and student encoder outputs
            combined_enc_output = ctnm_transformer.dynamic_switch(teacher_hidden_enc, student_enc_output)
            # Get student decoder output and prediction
            student_dec_output, student_prediction = self.ctn_transformer.get_student_decoder()(
                [tgt_inp, look_ahead_mask, nmt_mask, combined_enc_output])
            # Compute accuracy
            accuracy = self.ctn_transformer.train_accuracy(tgt_real, student_prediction)
            # Compute the loss between tgt_real and student_prediction
            nmt_loss_value = self.loss_object(tgt_real, student_prediction)
            # Compute NMT loss
            nmt_loss = self.ctn_transformer.train_loss(nmt_loss_value)
            # Compute distillation loss
            distillation_loss = mse_loss(teacher_hidden_enc, student_enc_output)
            # Compute total loss as a weighted sum of NMT and distillation loss
            combined_loss = nmt_loss * distillation_rate + distillation_loss * (1 - distillation_rate)

        all_trainable_variables = (self.ctn_transformer.transformer.encoder.trainable_variables +
                                   self.ctn_transformer.transformer.decoder.trainable_variables)

        nmt_gradients = tape.gradient(combined_loss, all_trainable_variables)

        # Get trainable variables for NMT model
        nmt_variables = self.ctn_transformer.transformer.encoder.trainable_variables + self.ctn_transformer.transformer.decoder.trainable_variables
        # Apply gradients to update the model parameters
        optimizer.apply_gradients(nmt_gradients, nmt_variables)
        # Return loss and accuracy
        return combined_loss, nmt_loss, distillation_loss, accuracy

    def train(self, dataset, epochs):
        # Initialize optimizer with learning rate
        optimizers = RateScheduledOptimizers(learning_rate_nmt=self.learning_rate)
        # Initialize step counter
        step_counter = 0
        # Iterate through epochs
        for epoch in range(epochs):
            # Reset epoch loss and accuracy
            self.ctn_transformer.train_loss.reset_states()
            self.ctn_transformer.train_accuracy.reset_states()
            # Iterate through dataset batches
            for batch in dataset:
                # Extract source and target language inputs
                (source_language_input_ids, target_language_input_ids) = batch

                distillation_rate = self.ctn_transformer.generate_distillation_rate(epoch, epochs)
                # Perform training step for the batch
                combined_loss, nmt_loss, distillation_loss, accuracy = self.train_step(source_language_input_ids,
                                                                                       target_language_input_ids,
                                                                                       optimizers, distillation_rate)
                # Print loss and accuracy for the epoch
                print(
                    f'Epoch {epoch + 1} Loss {self.ctn_transformer.train_loss.result():.4f} Accuracy '
                    f'{self.ctn_transformer.train_accuracy.result():.4f}')
                # Update learning rate
                lr = optimizers.optimizer_nmt.learning_rate

                # Increment step counter
                step_counter += 1
                # Record accuracies, losses, learning rates, and steps
                self.accuracies.append(accuracy.numpy())
                self.losses.append(nmt_loss.numpy())
                self.learning_rates.append(lr.numpy())
                self.steps.append(step_counter)

                # Save a checkpoint every 10 steps
                if step_counter % 5 == 0:
                    save_path = self.manager.save()
                    print(f"Saved checkpoint at step {step_counter}: {save_path}")

                    output_dir = 'outputs'
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    # Save accuracies, losses, learning rates, and steps to CSV file
                    df = pd.DataFrame({
                        'step': self.steps,
                        'accuracy': self.accuracies,
                        'loss': self.losses,
                        'learning_rate': self.learning_rates
                    })
                    # Now you can save the CSV file to the 'outputs' directory
                    df.to_csv(os.path.join(output_dir, 'visualization_data.csv'), index=False)

            # Generate graphs
            visualize_transformer_training(range(1, len(self.accuracies) + 1), self.accuracies, self.losses,
                                           self.source_language, self.target_language, "CTNMT Model")

            visualize_learningrate(range(1, len(self.accuracies) + 1), self.learning_rate,
                                   self.source_language, self.target_language, 'CTNMT Model')
