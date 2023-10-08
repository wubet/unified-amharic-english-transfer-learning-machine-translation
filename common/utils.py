import tensorflow as tf


def positional_encoding(position, d_model):
    angle_rates = 1 / tf.pow(10000, (2 * (tf.range(d_model, dtype=tf.float32) // 2)) / tf.cast(d_model, tf.float32))
    angle_rads = tf.reshape(angle_rates, (1, -1))
    angles = position * angle_rads

    sines = tf.math.sin(angles[:, 0::2])
    cosines = tf.math.cos(angles[:, 1::2])

    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = tf.expand_dims(pos_encoding, 0)

    return pos_encoding


def get_masks(x, y):
    encoder_padding_mask = create_padding_mask(x)
    decoder_padding_mask = create_padding_mask(x)
    look_ahead_mask = create_look_ahead_mask(tf.shape(y)[1])
    decoder_target_padding_mask = create_padding_mask(y)
    combined_mask = tf.maximum(
        decoder_target_padding_mask,
        look_ahead_mask
    )
    return encoder_padding_mask, combined_mask, decoder_padding_mask


def get_checkpoints(model, optimizer, checkpoint_dir, mode="train"):
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    if mode == "evaluate":
        if not manager.latest_checkpoint:
            raise ValueError("No checkpoint found. Cannot evaluate without a trained model.")
    elif mode == "train":
        if manager.latest_checkpoint:
            print("Restoring from", manager.latest_checkpoint)
            checkpoint.restore(manager.latest_checkpoint)
        else:
            print("No checkpoint found. Starting training from scratch.")

    return checkpoint, manager


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # add extra dimensions to match the BERT model's expected input


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


# def mse_loss(teacher_enc_output, student_enc_output):
#     return tf.reduce_mean(tf.square(teacher_enc_output - student_enc_output))
#
#


def loss_function(real, pried):
    # Masking the padding tokens (typically 0s in the sequence)
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    # Compute the initial loss values
    loss_ = loss_object(real, pried)

    # Convert the mask to the same data type as the loss values
    mask = tf.cast(mask, dtype=loss_.dtype)

    # Apply the mask to the loss values
    loss_ *= mask

    # Return the mean loss
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)
