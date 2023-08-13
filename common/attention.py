import tensorflow as tf


def ScaledDotProductAttention(q, k, v, mask):
    """
       Compute the scaled dot product attention.

       Parameters:
       ----------
       q : Tensor
           Query tensor of shape [batch_size, seq_length, d_model].

       k : Tensor
           Key tensor of shape [batch_size, seq_length, d_model].

       v : Tensor
           Value tensor of shape [batch_size, seq_length, d_model].

       mask : Tensor or None
           Mask tensor to prevent attention to certain positions.
           The shape should be broadcastable to [batch_size, num_heads, seq_length, seq_length].
           If None, no mask is applied.

       Returns:
       -------
       Tuple[Tensor, Tensor]
           Tuple containing the output tensor and the attention weights tensor.
    """
    qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    softmax_logits = qk / tf.math.sqrt(dk)
    if mask is not None:
        softmax_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(softmax_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights


def PositionWiseFFN(d_model, dff):
    """
        Create a position-wise feed forward network.

        Parameters:
        ----------
        d_model : int
            Depth of the model (often referred to as model's dimension).

        dff : int
            Depth of the feed-forward network's inner layer.

        Returns:
        -------
        tf.keras.Sequential
            A two-layer feed-forward network.
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])


class MultiHeadAttention(tf.keras.Model):
    """
        Multi-head attention mechanism.

        This class encapsulates the multi-head self-attention mechanism where the input is split
        into multiple heads which run in parallel and their outputs are concatenated
        before being linearly transformed into the final output.

        Attributes:
        ----------
        d_model : int
            Depth of the model.

        attention_heads : int
            Number of attention heads.

        depth : int
            Depth of individual attention head. Computed as d_model divided by attention_heads.
    """

    def __init__(self, d_model, attention_heads):
        """
            Initialize the multi-head attention mechanism.

            Parameters:
            ----------
            d_model : int
                Depth of the model.

            attention_heads : int
                Number of attention heads.
        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.attention_heads = attention_heads
        self.depth = self.d_model // self.attention_heads
        self.linear_q = tf.keras.layers.Dense(self.d_model)
        self.linear_k = tf.keras.layers.Dense(self.d_model)
        self.linear_v = tf.keras.layers.Dense(self.d_model)
        self.linear = tf.keras.layers.Dense(self.d_model)

    def split_heads(self, x, batch_size):
        """
            Split the input tensor into multiple heads.

            Parameters:
            ----------
            x : Tensor
                Input tensor of shape [batch_size, seq_length, d_model].

            batch_size : int
                Number of samples in the batch.

            Returns:
            -------
            Tensor
                Tensor reshaped to [batch_size, attention_heads, seq_length, depth].
        """
        x = tf.reshape(x, (batch_size, -1, self.attention_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        """
            Perform multi-head attention and produce the output.

            Parameters:
            ----------
            v : Tensor
                Value tensor of shape [batch_size, seq_length, d_model].

            k : Tensor
                Key tensor of shape [batch_size, seq_length, d_model].

            q : Tensor
                Query tensor of shape [batch_size, seq_length, d_model].

            mask : Tensor or None
                Mask tensor to prevent attention to certain positions.
                If None, no mask is applied.

            Returns:
            -------
            Tuple[Tensor, Tensor]
                Tuple containing the output tensor and the attention weights tensor.
        """
        batch_size = tf.shape(q)[0]
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, attention_weights = ScaledDotProductAttention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.linear(concat_attention)
        return output, attention_weights
