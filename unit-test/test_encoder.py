from unittest import TestCase
import tensorflow as tf
from encoders.encoder import Encoder
from layers.encoder_layer import EncoderLayer


class TestEncoder(TestCase):
    def test_call(self):
        self.fail()

    def test_encoder_layer(self):
        print("\nTesting EncoderLayer...")
        sample_encoder_layer = EncoderLayer(
            d_model=512,
            num_heads=8,
            dff=2048
        )

        sample_encoder_layer_output = sample_encoder_layer(
            tf.random.uniform((64, 43, 512)),  # (batch_size, input_seq_len, d_model)
            False,
            None
        )

        print(sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)
        assert sample_encoder_layer_output.shape == (64, 43, 512), "Unexpected shape for EncoderLayer output!"

    def test_encoder(self):
        print("\nTesting Encoder...")
        sample_encoder = Encoder(
            num_layers=2,
            d_model=512,
            num_heads=8,
            dff=2048,
            input_vocab_size=8500
        )

        sample_encoder_output = sample_encoder(
            tf.random.uniform((64, 38)),  # (batch_size, input_seq_len)
            training=False,
            mask=None
        )

        print(sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)
        assert sample_encoder_output.shape == (64, 38, 512), "Unexpected shape for Encoder output!"

    if __name__ == "__main__":
        test_encoder_layer()
        test_encoder()
