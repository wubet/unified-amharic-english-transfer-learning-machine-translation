"""Utility functions and classes. """
import heapq

import numpy as np
import tensorflow as tf


class CosineDecayLearningRateSchedule(
    tf.keras.optimizers.schedules.LearningRateSchedule):
  """Defines cosine decay learning rate."""
  def __init__(
      self, learning_rate, decay_steps, alpha, warmup_steps, warmup_lr):
    """Constructor.

    Args:
      learning_rate: float scalar, the base learning rate.
      decay_steps: int scalar, num of steps to decay over.
      alpha: float scalar, minimum learning rate value as a fraction of
        learning rate.
      warmup_steps: int scalar, the num of warm-up steps.
      warmup_lr: float scalar, learning rate for warm-up steps.
    """
    super(CosineDecayLearningRateSchedule, self).__init__()
    self._learning_rate = learning_rate
    self._decay_steps = decay_steps
    self._alpha = alpha
    self._warmup_steps = warmup_steps
    self._warmup_lr = warmup_lr

  def __call__(self, global_step):
    """Computes learning rate.

    Args:
      global_step: int scalar tensor, the current global step.

    Returns:
      learning_rate: float scalar tensor, the learning rate as a function of
        the input `global_step`.
    """
    global_step = tf.cast(global_step, 'float32')

    cosine_decay = 0.5 * (1 + tf.cos(np.pi * tf.minimum(global_step
        - self._warmup_steps, self._decay_steps) / self._decay_steps))
    decayed = (1 - self._alpha) * cosine_decay + self._alpha
    decayed_learning_rate = self._learning_rate * decayed

    decayed_learning_rate = tf.where(global_step < self._warmup_steps,
                                     self._warmup_lr,
                                     decayed_learning_rate)

    return decayed_learning_rate


class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Learning rate schedule."""
  def __init__(self, learning_rate, hidden_size, warmup_steps):
    """Constructor.

    Args:
      learning_rate: float scalar, the base learning rate.
      hidden_size: int scalar, the hidden size of continuous representation.
      warmup_steps: int scalar, the num of warm-up steps
    """
    super(LearningRateSchedule, self).__init__()
    self._learning_rate = learning_rate
    self._hidden_size = hidden_size
    self._warmup_steps = tf.cast(warmup_steps, 'float32')

  def __call__(self, global_step):
    """Computes learning rate with linear warmup and rsqrt decay.

    Args:
      global_step: int scalar tensor, the current global step.

    Returns:
      learning_rate: float scalar tensor, the learning rate as a function of
        the input `global_step`.
    """
    global_step = tf.cast(global_step, 'float32')
    learning_rate = self._learning_rate
    learning_rate *= (self._hidden_size**-0.5)
    # linear warmup
    learning_rate *= tf.minimum(1.0, global_step / self._warmup_steps)
    # rsqrt decay
    learning_rate /= tf.sqrt(tf.maximum(global_step, self._warmup_steps))
    return learning_rate


def save_attention_weights(filename, data):
  """Saves attention weights data to *.npy file.

  Args:
    filename: string scalar, filename.
    data: a list or tuple or dict of numpy arrays, the attention weights and
      token ids of input and translated sequence.
  """
  np.save(filename, data)


def dict_to_example(dictionary):
  """Convert dict to protobuf example message.

  Args:
    dictionary: a dict mapping string to list of integers

  Returns:
    a protobuf example message.
  """
  features = {}
  for k, v in dictionary.items():
    features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
  return tf.train.Example(features=tf.train.Features(feature=features))


def nucleus_sampling(scores, threshold=0.95):
  """Sample from the head of the probability distribution that contains the
  vast majority of probability mass. See https://arxiv.org/abs/1904.09751
  for details. The distribution is truncated to the  and re-normalized.

  Args:
    scores: numpy array of shape [vocab_size], the probability distribution (
      sum to one) of all possible next-tokens over the vocabulary.
    threshold: float scalar, the minimum value of the sum of probability mass
      that the head of the distribution must exceed.

  Returns:
    next_token_id: int scalar, the sampled id of the next token.
  """
  ids = np.argsort(-scores)
  cumsum = [0.] + np.cumsum(scores[ids]).tolist()
  # search space is any value >= low and <= high
  low, high = 0, len(cumsum) - 2

  while low <= high:
    mid = (low + high) // 2
    sum1 = cumsum[mid]
    sum2 = cumsum[mid + 1]
    if sum1 < threshold and sum2 >= threshold:
      break
    elif sum2 < threshold: # exclude indices <= mid
      low = mid + 1
    elif sum1 >= threshold: # exclude indices >= mid
      high = mid - 1
    else:
      raise ValueError('Impossible outcome')

  probs = scores[ids[:mid + 1]] / sum2
  next_token_id = np.random.choice(ids[:mid + 1], p=probs)
  return next_token_id


def topk_sampling(scores, k=40):
  """Sample from the top-k tokens with the largest probability. The distribution
   is truncated and re-normalized.

  Args:
    scores: numpy array of shape [vocab_size], the probability distribution (
      sum to one) of all possible next-tokens over the vocabulary.
    k: int scalar, the num of next-tokens with largest probability to sample
      from.

  Returns:
    next_token_id: int scalar, the sampled id of the next token.
  """
  min_pq = list(zip(scores[:k], range(k)))
  heapq.heapify(min_pq)
  for i in np.arange(k, len(scores)):
    if scores[i] > min_pq[0][0]:
      min_pq[0] = scores[i], i
      heapq.heapify(min_pq)

  probs, ids = list(zip(*min_pq))
  probs = np.array(probs)
  probs /= probs.sum()
  next_token_id = np.random.choice(ids, p=probs)
  return next_token_id


def rel_shift(inputs):
  """Shift the matrix in the input tensor, so that the query position matches
  correctly with the key position for computing attention scores.

  Given input tensor `x` of shape [batch_size, num_heads, q_seq_len, r_seq_len],
  each slice `x[i, j]` is a matrix of shape [q_seq_len, r_seq_len] (Note that
  generally `r_seq_len` >= `q_seq_len`

  the matrix `x[i, j]` in the output will be a left-shifted version of the input
  , where the 0th, 1st, ..., and `q_seq_len - 1`-th row will be left-shifted by
  `q_seq_len - 1`, `q_seq_len - 2`, ..., and 0 positions.


  Args:
    inputs: float tensor of shape [batch_size, num_heads, q_seq_len, r_seq_len],
      the input tensor.

  Returns:
    outputs: float tensor of shape [batch_size, num_heads, q_seq_len, r_seq_len]
      , the shifted tensor.
  """
  shape = tf.shape(inputs)
  padded = tf.pad(inputs, [[0, 0], [0, 0], [0, 0], [1, 0]])
  reshaped = tf.reshape(padded, [shape[0], shape[1], shape[3] + 1, shape[2]])
  sliced = reshaped[:, :, 1:]
  outputs = tf.reshape(sliced, shape)
  return outputs

# def load_and_tokenize(self):
#     # Load the source and target files
#     src_dataset = tf.data.TextLineDataset(self.src_file_path)
#     tgt_dataset = tf.data.TextLineDataset(self.tgt_file_path)
#
#     # Tokenize the sentences
#     src_dataset = src_dataset.map(self.tokenize_sentences)
#     tgt_dataset = tgt_dataset.map(self.tokenize_sentences)
#
#     # Pad the sequences
#     src_dataset = src_dataset.map(lambda x, y: pad_sequences(x, padding='post'))
#     tgt_dataset = tgt_dataset.map(lambda x, y: pad_sequences(x, padding='post'))
#
#     # Zip the datasets together
#     train_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
#
#     # Batch the dataset
#     train_dataset = train_dataset.batch(self.batch_size)
#
#     return train_dataset

# def load_and_tokenize(self):
#     # Load the source and target files
#     src_dataset = tf.data.TextLineDataset(self.src_file_path)
#     tgt_dataset = tf.data.TextLineDataset(self.tgt_file_path)
#
#     # Tokenize the sentences
#     src_dataset = src_dataset.map(self.tokenize_sentences)
#     tgt_dataset = tgt_dataset.map(self.tokenize_sentences)
#
#     # Flatten the datasets and separate input_ids and attention_mask
#     src_dataset = src_dataset.map(lambda features: (features['input_ids'], features['attention_mask']))
#     tgt_dataset = tgt_dataset.map(lambda features: (features['input_ids'], features['attention_mask']))
#
#     # Zip the datasets together
#     train_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
#
#     # Batch the dataset with padding
#     train_dataset = train_dataset.padded_batch(self.batch_size, padded_shapes=(([None], [None]), ([None], [None])))
#
#     return train_dataset

# def load_and_tokenize(self):
#     # Load the source and target files
#     src_dataset = tf.data.TextLineDataset(self.src_file_path)
#     tgt_dataset = tf.data.TextLineDataset(self.tgt_file_path)
#
#     # Tokenize the sentences
#     src_dataset = src_dataset.map(self.tokenize_sentences)
#     tgt_dataset = tgt_dataset.map(self.tokenize_sentences)
#
#     # Flatten the datasets and separate input_ids and attention_mask
#     src_dataset = src_dataset.map(lambda features: (features['input_ids'], features['attention_mask']))
#     tgt_dataset = tgt_dataset.map(lambda features: (features['input_ids'], features['attention_mask']))
#
#     # Zip the datasets together
#     train_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
#
#     # Function to calculate the length of an item
#     def element_length_func(x, y):
#         return tf.shape(x)[0]
#
#     # Boundaries for the buckets
#     boundaries = [50, 100, 150, 200, 250]  # Modify these values as needed
#
#     # Batch sizes for each bucket
#     batch_sizes = [self.batch_size] * (len(boundaries) + 1)
#
#     # Use bucket_by_sequence_length to bucket the sentences
#     train_dataset = train_dataset.apply(
#         tf.data.experimental.bucket_by_sequence_length(
#             element_length_func, boundaries, batch_sizes, padded_shapes=(([None], [None]), ([None], [None]))
#         )
#     )
#
#     # # Find the maximum sequence length across both the source and target datasets
#     # max_length = max(src_dataset.map(lambda x, y: tf.shape(x)[0]).reduce(tf.constant(0), tf.maximum),
#     #                  tgt_dataset.map(lambda x, y: tf.shape(x)[0]).reduce(tf.constant(0), tf.maximum))
#     #
#     # # Pad both the source and target datasets to the maximum length
#     # train_dataset = train_dataset.map(lambda src, tgt: ((tf.pad(src[0], [0, max_length - tf.shape(src[0])[0]]),
#     #                                                      tf.pad(src[1], [0, max_length - tf.shape(src[1])[0]])),
#     #                                                     (tf.pad(tgt[0], [0, max_length - tf.shape(tgt[0])[0]]),
#     #                                                      tf.pad(tgt[1], [0, max_length - tf.shape(tgt[1])[0]]))))
#
#     return train_dataset

 # def distillation_loss(self, labels, predictions, teacher_predictions, temperature=2.0, alpha=0.1):
    #     # Asymptotic distillation loss with teacher and student predictions
    #     teacher_predictions = tf.stop_gradient(teacher_predictions)
    #
    #     # Compute the cross-entropy loss between student predictions and labels
    #     student_loss = self.loss_object(labels, predictions)
    #
    #     # Compute the cross-entropy loss between teacher predictions and labels
    #     teacher_loss = self.loss_object(labels, teacher_predictions)
    #
    #     # Apply temperature scaling to both student and teacher predictions
    #     scaled_student_predictions = predictions / temperature
    #     scaled_teacher_predictions = teacher_predictions / temperature
    #
    #     # Compute the softmax cross-entropy loss between the scaled student and teacher predictions
    #     kd_loss = self.loss_object(scaled_teacher_predictions, scaled_student_predictions)
    #
    #     # Compute the final distillation loss as a weighted sum of the student loss and the knowledge distillation loss
    #     distillation_loss = alpha * student_loss + (1 - alpha) * kd_loss
    #
    #     return distillation_loss

  # class TrainTransferTransformer:
  #   def __init__(self, transformer, optimizer, loss_object, train_loss, train_accuracy, checkpoint_path):
  #     self.transformer = transformer
  #     self.optimizer = optimizer
  #     self.loss_object = loss_object
  #     self.train_loss = train_loss
  #     self.train_accuracy = train_accuracy
  #     # "./checkpoints/ckpt"
  #     self.checkpoint_path = checkpoint_path
  #
  #   def create_masks(self, inp, tar):
  #     encoder_padding_mask = self.transformer.create_padding_mask(inp)
  #     decoder_padding_mask = self.transformer.create_padding_mask(inp)
  #     look_ahead_mask = self.transformer.create_look_ahead_mask(tf.shape(tar)[1])
  #     dec_target_padding_mask = self.transformer.create_padding_mask(tar)
  #     combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
  #     return encoder_padding_mask, combined_mask, decoder_padding_mask
  #
  #   def loss_function(self, real, pred):
  #     mask = tf.math.logical_not(tf.math.equal(real, 0))
  #     loss = self.loss_object(real, pred)
  #
  #     mask = tf.cast(mask, dtype=loss.dtype)
  #     loss *= mask
  #
  #     return tf.reduce_mean(loss)
  #
  #   def train_step(self, inp, tar, tchr=None, use_distillation=False):
  #     tar_inp = tar[:, :-1]
  #     tar_real = tar[:, 1:]
  #
  #     encoder_padding_mask, combined_mask, decoder_padding_mask = self.create_masks(inp, tar_inp)
  #
  #     with tf.GradientTape() as tape:
  #       predictions, _ = self.transformer(
  #         inp, tar_inp, True, encoder_padding_mask, combined_mask, decoder_padding_mask
  #       )
  #
  #       if use_distillation and tchr is not None:
  #         tchr_predictions, _ = tchr.transformer(
  #           inp, tar_inp, True, encoder_padding_mask, combined_mask, decoder_padding_mask
  #         )
  #         distillation_loss = self.loss_object(tar_real, tchr_predictions)
  #         loss = distillation_loss + self.loss_object(tar_real, predictions)
  #       else:
  #         loss = self.loss_object(tar_real, predictions)
  #
  #     gradients = tape.gradient(loss, self.transformer.trainable_variables)
  #     self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))
  #
  #     self.train_loss(loss)
  #     self.train_accuracy(tar_real, predictions)
  #
  #   def train(self, train_dataset, teacher=None, use_distillation=False, epochs=20, rate_schedule=False):
  #     for epoch in range(epochs):
  #       self.train_loss.reset_states()
  #       self.train_accuracy.reset_states()
  #
  #       if rate_schedule:
  #         self.optimizer.lr.assign(self.rate_scheduler(epoch))
  #
  #       for (batch, (inp, tar)) in enumerate(train_dataset):
  #         self.train_step(inp, tar, teacher, use_distillation)
  #
  #       if (epoch + 1) % 5 == 0:
  #         ckpt_save_path = self.checkpoint_path
  #         self.transformer.save_weights(ckpt_save_path)
  #         print(f"Saving checkpoint for epoch {epoch + 1}")
  #         print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')
  #
  #       print(
  #         f"Epoch {epoch + 1} Loss {self.train_loss.result():.4f} Accuracy {self.train_accuracy.result():.4f}"
  #       )
  #
  #   def rate_scheduler(self, epoch, warmup_steps=4000):
  #     arg1 = tf.math.rsqrt(tf.cast(epoch + 1, tf.float32))
  #     arg2 = epoch * (warmup_steps ** -1.5)
  #     return tf.math.rsqrt(tf.cast(self.transformer.d_model, tf.float32)) * tf.math.minimum(arg1, arg2)

