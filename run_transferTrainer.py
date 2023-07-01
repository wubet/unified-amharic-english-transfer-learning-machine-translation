import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from learningTransformer import TransferLearningTransformer
from transfer_model_runner import TrainTransferTransformer, TrainTransferLearningTransformer


# Assuming we are using a preprocessed TensorFlow dataset
train_dataset_path = "path_to_your_train_dataset"
train_dataset = tf.data.experimental.load(train_dataset_path)

# Define the hypermarkets
d_model = 512
num_layers = 6
num_heads = 8
dff = 2048
dropout_rate = 0.1
learning_rate = 2.0

# Load the teacher model
teacher_model_path = "bert-base-uncased" # Path to the pretrained model

# Load the dataset to calculate the vocab sizes
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    (data.numpy() for data, _ in train_dataset), target_vocab_size=2**13)

# Compute the vocab sizes. +2 for <start> and <end> tokens
input_vocab_size = target_vocab_size = tokenizer.vocab_size + 2


# # Instantiate the TransferLearningTransformer class
# train_nmt = TransferLearningTransformer(input_vocab_size, target_vocab_size, d_model, num_layers, num_heads, dff, dropout_rate, teacher_model_path)
#
# # Create the Transformer model and optimizer
# train_nmt.create_transformer()
# train_nmt.create_optimizer(learning_rate)
#
# # Create metrics for monitoring training
# train_nmt.create_metrics()
#
# # Define the checkpoint directory and file path
# checkpoint_dir = '/path/to/checkpoint_directory'
# checkpoint_path = checkpoint_dir + '/cp-{epoch:04d}.ckpt'
#
# # Define your training dataset
# train_dataset_path = "path_to_your_train_dataset"
# train_dataset = tf.data.experimental.load(train_dataset_path)
#
# # Create an instance of the TrainTransferTransformer class
# training_process = TrainTransferTransformer(train_nmt, learning_rate, checkpoint_path)
#
# # Start the training
# training_process.train(train_dataset, epochs=10)


# Initialize the transformer model for transfer learning
transfer_learning_transformer = TransferLearningTransformer(
    input_vocab_size=input_vocab_size,
    target_vocab_size=target_vocab_size,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    dff=dff,
    dropout_rate=dropout_rate,
    teacher_model_path=teacher_model_path)

# Create the Transformer model
transfer_learning_transformer.create_transformer()

# Create the optimizer and loss object
transfer_learning_transformer.create_optimizer(learning_rate=learning_rate)

# Create the metrics for monitoring training
transfer_learning_transformer.create_metrics()

# Initialize the training class with the transformer model
train_transfer_learning_transformer = TrainTransferLearningTransformer(transfer_learning_transformer)

# Assuming `dataset` is your tf.data.Dataset object with inputs and targets
epochs = 10 # Set your desired number of epochs
train_transfer_learning_transformer.train(train_dataset, epochs)


# # Define the checkpoint callback to save the model weights during training
# checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True)
#
# # Define the learning rate schedule callback
# lr_schedule_callback = LearningRateScheduler(train_nmt.learning_rate_schedule)
#
# # Define the number of epochs for training
# epochs = 10
#
# # Train the teacher-student model
# for epoch in range(epochs):
#     train_nmt.train_loss.reset_states()
#     train_nmt.train_accuracy.reset_states()
#
#     # Iterate through the training dataset
#     for (inputs, targets) in train_dataset:
#         # Compute teacher predictions using the BERT model
#         teacher_predictions = train_nmt.teacher_transformer(inputs)
#
#         with tf.GradientTape() as tape:
#             # Compute student predictions using the Transformer model
#             predictions = train_nmt.transformer_model([inputs, targets[:, :-1]])
#
#             # Compute distillation loss
#             loss = train_nmt.distillation_loss(targets[:, 1:], predictions, teacher_predictions)
#
#         # Compute gradients and update weights
#         gradients = tape.gradient(loss, train_nmt.transformer_model.trainable_variables)
#         train_nmt.optimizer.apply_gradients(zip(gradients, train_nmt.transformer_model.trainable_variables))
#
#         # Update training metrics
#         train_nmt.train_loss(loss)
#         train_nmt.train_accuracy(targets[:, 1:], predictions)
#
#     # Print training metrics for the epoch
#     print(f'Epoch {epoch + 1}: Loss = {train_nmt.train_loss.result()}, Accuracy = {train_nmt.train_accuracy.result()}')
#
#     # Save the model weights at the end of each epoch
#     train_nmt.transformer_model.save_weights(checkpoint_path.format(epoch=epoch))
#
# # Restore the model from the last checkpoint
# latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
# train_nmt.transformer_model.load_weights(latest_checkpoint)
