import tensorflow as tf
from tensorflow.keras import layers
from transformers.models.ctrl.modeling_ctrl import point_wise_feed_forward_network


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model,  dropout=dropout_rate)
        # print("MHA", self.mha)
        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        # self.dropout1 = layers.Dropout(dropout_rate)
        # self.dropout2 = layers.Dropout(dropout_rate)
        self.dropout_ffn = layers.Dropout(dropout_rate)

        # self.mha = layers.MultiHeadAttention(d_model, num_heads)
        # # print("after MHA", self.mha)
        # self.ffn = point_wise_feed_forward_network(d_model, dff)
        # # print("ffn", self.ffn)
        #
        # self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        #
        # self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        # self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        """
        Forward pass for the Encoder Layer
        :param x:
            x -- Tensor of shape (batch_size, input_seq_len, fully_connected_dim)
        :param training:
            Boolean, set to true to activate the training mode for dropout layers
        :param mask:
            Boolean mask to ensure that the padding is not treated as part of the input
        :return:
            encoder_layer_out -- Tensor of shape (batch_size, input_seq_len, fully_connected_dim)
            attn_weights -- attention weights from MultiHeadAttention layer
        """
        # tf.config.run_functions_eagerly(True)
        # calculate self-attention using mha(~1 line). Dropout will be applied during training
        # print("Before MHA: ", x.shape)
        # outputs = self.mha(query=x, key=x, value=x, attention_mask=mask)  # (batch_size,
        # # input_seq_len, d_model)
        # attn_output = outputs[0]
        # attn_weights = outputs[1]
        # print("After MHA: ", x.shape)
        # attn_output = self.dropout1(attn_output, training=training)
        # out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        # ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        # ffn_output = self.dropout2(ffn_output, training=training)
        # out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        #
        # return out2, attn_weights
        # print("MHA", "Start of MHA")
        # attn_output, _ = self.mha(x, x, x, mask)  # Self attention (encoder looks at its own input)
        # print("MHA", attn_output)
        # attn_output = self.dropout1(attn_output, training=training)
        # out1 = self.layernorm1(x + attn_output)  # Add & Normalize
        #
        # ffn_output = self.ffn(out1)  # Feed Forward
        # ffn_output = self.dropout2(ffn_output, training=training)
        # out2 = self.layernorm2(out1 + ffn_output)  # Add & Normalize
        # print("encoder_layer_out", out2.shape)
        # return out2
        print("MHA", "Start of MHA")
        attn_output = self.mha(query=x,
                               value=x,
                               key=x,
                               attention_mask=mask,
                               training=training)  # Self attention (batch_size, input_seq_len, fully_connected_dim)
        print("MHA After", attn_output)
        # 前面的hints里又说training为默认值，默认值即为true所以这里删去training也可以，所以这里实际已经droupout了不需要单独的dense层在执行
        # apply layer normalization on sum of the input and the attention output to get the
        # output of the multi-head attention layer (~1 line)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, fully_connected_dim)

        # pass the output of the multi-head attention layer through a ffn (~1 line)
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, fully_connected_dim)

        # apply dropout layer to ffn output during training (~1 line)
        ffn_output = self.dropout_ffn(ffn_output, training)

        # apply layer normalization on sum of the output from multi-head attention and ffn output to get the
        # output of the encoder layer (~1 line)
        encoder_layer_out = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, fully_connected_dim)
        # END CODE HERE
        print("encoder_layer_out: ", encoder_layer_out)
        return encoder_layer_out

        # calculate self-attention using mha(~1 line). Dropout will be applied during training
        # attn_output = self.mha(query=x,
        #                        value=x,
        #                        key=x,
        #                        attention_mask=mask,
        #                        training=training)  # Self attention (batch_size, input_seq_len, fully_connected_dim)
        # # 前面的hints里又说training为默认值，默认值即为true所以这里删去training也可以，所以这里实际已经droupout了不需要单独的dense层在执行
        # # apply layer normalization on sum of the input and the attention output to get the
        # # output of the multi-head attention layer (~1 line)
        # out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, fully_connected_dim)
        #
        # # pass the output of the multi-head attention layer through a ffn (~1 line)
        # ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, fully_connected_dim)
        #
        # # apply dropout layer to ffn output during training (~1 line)
        # ffn_output = self.dropout_ffn(ffn_output, training)
        #
        # # apply layer normalization on sum of the output from multi-head attention and ffn output to get the
        # # output of the encoder layer (~1 line)
        # encoder_layer_out = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, fully_connected_dim)
        # # END CODE HERE
        #
        # return encoder_layer_out