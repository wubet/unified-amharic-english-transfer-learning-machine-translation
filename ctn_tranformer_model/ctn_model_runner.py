from time import time
from common.rate_scheduler import RateScheduledOptimizers
from common.utils import *
import os
import itertools
import pandas as pd
from common.visualization import visualize_transformer_training, visualize_learningrate
from ctn_tranformer_model.ctnmt import CTNMTransformer
from ctn_tranformer_model.custom_train_model import CustomTrainingModel


class TrainCtnmtModel:
    """
        A class used to train a custom NMT model using knowledge distillation.

        Attributes:
        ----------
        transformer : Transformer model
            The student model that we're trying to train.
        teacher_model : Model
            The teacher model from which knowledge is being transferred to the student model.
        checkpoint_dir : str
            Directory to save and load model checkpoints.
        source_language : str
            The language of the source data.
        target_language : str
            The language of the target data.
        learning_rate : float
            Learning rate for training the model.
        accuracies : list or None
            List to store accuracy values during training.
        losses : list or None
            List to store loss values during training.
        learning_rates : list or None
            List to store learning rate values during training.
        steps : list or None
            List to store the step numbers during training.
        ctnmt : CTNMTransformer
            CTNMT model instance.
        optimizers : RateScheduledOptimizers
            Optimizer used for training.
        custom_model : CustomTrainingModel
            Custom training model combining student and teacher models.
        train_loss : tf.keras.metrics.Mean
            Metric for tracking the training loss.
        train_accuracy : tf.keras.metrics.SparseCategoricalAccuracy
            Metric for tracking the training accuracy.
        summary_writer : tf.summary.create_file_writer
            Writer for logging training metrics.
        num_iterations : int
            The number of iterations for training.
        persist_per_iterations : int
            The number of iterations to perform before persisting checkpoints.
        log_per_iterations : int
            The number of iterations to perform before logging training metrics.
    """

    def __init__(self, transformer, teacher_model, checkpoint_dir,
                 learning_rate, d_model, num_iterations, persist_per_iterations,
                 log_per_iterations, source_language, target_language):
        """
            Constructor to initialize the training model.

            Parameters:
            ----------
            transformer : Transformer model
                The student model that we're trying to train.
            teacher_model : Model
                The teacher model from which knowledge is being transferred.
            checkpoint_dir : str
                Directory to save and load checkpoints.
            learning_rate : float
                Learning rate for the model.
            d_model : int
                The dimensionality of the output space (i.e. hidden connection network) for the model.
            num_iterations : int
                The number of iterations for training.
            persist_per_iterations : int
                The number of iterations to perform before persisting checkpoints.
            log_per_iterations : int
                The number of iterations to perform before logging training metrics.
            source_language : str
                Language of the source data.
            target_language : str
                Language of the target data.
        """
        self.transformer = transformer
        self.teacher_model = teacher_model
        self.checkpoint_dir = checkpoint_dir
        self.num_iterations = num_iterations
        self.persist_per_iterations = persist_per_iterations
        self.log_per_iterations = log_per_iterations
        self.source_language = source_language
        self.target_language = target_language
        self.learning_rate = learning_rate
        self.accuracies = None
        self.losses = None
        self.learning_rates = None
        self.steps = None
        self.ctnmt = CTNMTransformer(d_model)
        self.optimizers = RateScheduledOptimizers(learning_rate_nmt=self.learning_rate)
        self.custom_model = CustomTrainingModel(self.transformer, self.teacher_model,
                                                self.ctnmt)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.summary_writer = tf.summary.create_file_writer('logs')

    def train(self, dataset):
        """
           Trains the CTNMT model using the given dataset and number of epochs.

           This method:
           - Initializes metrics for training (loss and accuracy).
           - Sets up TensorFlow summary writers for logging.
           - Loads checkpoints if available, or initializes training metrics if no checkpoints are found.
           - Defines the inner training step function.
           - Iterates through each epoch and batch, performing forward and backward passes.
           - Optionally saves checkpoints and logs metrics.
           - Visualizes training metrics after training completion.

           Parameters:
           ----------
           dataset : tf.data.Dataset
               The training dataset, expected to yield batches of source and target sequences.

           Returns:
           -------
           None
        """

        # Path for saving visualization data as CSV
        csv_path = 'outputs/visualization_data.csv'
        checkpoint, checkpoint_manager = get_checkpoints(self.transformer, self.optimizers.optimizer,
                                                         self.checkpoint_dir)
        output_dir = 'outputs'

        # If a checkpoint exists, restore from it and load metrics
        if checkpoint_manager.latest_checkpoint:
            checkpoint.restore(checkpoint_manager.latest_checkpoint)
            print("Restored from checkpoint:", checkpoint_manager.latest_checkpoint)
            # Load previous accuracy, loss, learning rates, and steps from CSV file
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
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

        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.float32)
            ]
        )
        def train_step(src, tgt, distillation_rate):
            target_input = tgt[:, :-1]
            target_real = tgt[:, 1:]
            encoder_padding_mask, combined_mask, decoder_padding_mask = get_masks(src, target_input)
            with tf.GradientTape() as tape:
                student_prediction, student_enc_output, teacher_hidden_enc = self.custom_model(src, target_input,
                                                                                               encoder_padding_mask,
                                                                                               combined_mask,
                                                                                               decoder_padding_mask)

                loss = loss_function(target_real, student_prediction)
                nmt_accuracy = self.train_accuracy(target_real, student_prediction)
                # Compute distillation loss
                distillation_loss = self.ctnmt.asymptotic_distillation(teacher_hidden_enc, student_enc_output)
                # Compute total loss as a weighted sum of NMT and distillation loss
                combined_loss = loss * distillation_rate + distillation_loss * (1 - distillation_rate)

            nmt_gradients = tape.gradient(combined_loss, self.transformer.trainable_variables)
            # Apply gradients to update the model parameters
            self.optimizers.apply_gradients(nmt_gradients, self.transformer.trainable_variables)
            lr = self.optimizers.optimizer.learning_rate
            step = self.optimizers.optimizer.iterations
            return self.train_loss(combined_loss), nmt_accuracy, lr, step

        step = 0
        start_time = time()
        # Create a cycle of your dataset
        dataset_cycle = itertools.cycle(dataset)

        for i in range(self.num_iterations):
            # Get the next batch of data from the cycle
            source, target = next(dataset_cycle)

            distillation_rate = self.ctnmt.generate_distillation_rate(step, self.num_iterations)
            batch_loss, batch_accuracy, lr, step = train_step(source, target, distillation_rate)
            with self.summary_writer.as_default():
                tf.summary.scalar('train_loss', batch_loss, step=step)
                tf.summary.scalar('learning_rate', lr, step=step)

            if step.numpy() % self.log_per_iterations == 0:
                print('global step: %d, loss: %f, accuracy: %f, learning rate:' %
                      (step.numpy(), batch_loss.numpy(), self.train_accuracy.result()), float(lr.numpy()))
                self.steps.append(step.numpy())
                self.losses.append(batch_loss.numpy())
                self.accuracies.append(tf.keras.backend.get_value(self.train_accuracy.result()))
                self.learning_rates.append(float(lr.numpy()))

            if step.numpy() % self.persist_per_iterations == 0:
                save_path = checkpoint_manager.save()
                print(f"Saved checkpoint at step {step}: {save_path}")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                df = pd.DataFrame({
                    'step': self.steps,
                    'accuracy': self.accuracies,
                    'loss': self.losses,
                    'learning_rate': self.learning_rates
                })
                # Now you can save the CSV file to the 'outputs' directory
                df.to_csv(os.path.join(output_dir, 'visualization_data.csv'), index=False)
                print(
                    f"Saved training data at step {step}: {os.path.join(output_dir, 'visualization_data.csv')}")

            if step.numpy() == self.num_iterations:
                f = open(os.path.join(output_dir, "visualization_data.csv"), "w")
                f.truncate()
                f.close()
                break

        print('Done. Time taken: {} seconds'.format(time() - start_time))
        # Generate graphs
        visualize_transformer_training(range(1, len(self.accuracies) + 1), self.accuracies, self.losses,
                                       self.source_language, self.target_language, "CTNMT_Model")
        visualize_learningrate(range(1, len(self.accuracies) + 1), self.learning_rates,
                               self.source_language, self.target_language, 'CTNMT_Model')
