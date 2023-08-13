import tensorflow as tf


class CustomTrainingModel(tf.keras.Model):
    def __init__(self, transformer, teacher_model, ctnmt, **kwargs):
        super(CustomTrainingModel, self).__init__(**kwargs)
        self.transformer = transformer
        self.teacher_model = teacher_model
        self.ctnmt = ctnmt

    def call(self, src, target_input, encoder_padding_mask, combined_mask, decoder_padding_mask):
        teacher_enc_output = self.teacher_model(src, training=False)
        teacher_hidden_enc = None
        if 'last_hidden_state' in teacher_enc_output:
            teacher_hidden_enc = teacher_enc_output['last_hidden_state']

        student_enc_output = self.transformer.encoder(src, True, encoder_padding_mask)
        combined_enc_output = self.ctnmt.dynamic_switch(teacher_hidden_enc, student_enc_output)

        student_decoder_output, _ = self.transformer.decoder(
            target_input, combined_enc_output, True, combined_mask, decoder_padding_mask
        )
        # Apply the final layer of the transformer to get predictions
        # student_prediction = self.transformer.final_layer

        # student_prediction, _ = self.transformer(
        #     src, target_input, True, encoder_padding_mask, combined_mask, decoder_padding_mask
        # )
        # Use the Transformer's call method directly with combined_enc_output as the external encoder output
        student_prediction, _ = self.transformer(
            src, target_input, True, encoder_padding_mask, combined_mask, decoder_padding_mask, combined_enc_output
        )

        return student_prediction, student_enc_output, teacher_hidden_enc
        # return student_prediction

    # def call(self, src, target_input, encoder_padding_mask, combined_mask, decoder_padding_mask):
    #     # teacher_enc_output = self.teacher_model(src, training=False)
    #     # teacher_hidden_enc = None
    #     # if 'last_hidden_state' in teacher_enc_output:
    #     #     teacher_hidden_enc = teacher_enc_output['last_hidden_state']
    #
    #     # student_enc_output = self.transformer.encoder(src, True, encoder_padding_mask)
    #     # # combined_enc_output = self.ctnmt.dynamic_switch(teacher_hidden_enc, student_enc_output)
    #     #
    #     # student_decoder_output, _ = self.transformer.decoder(
    #     #     target_input, student_enc_output, True, combined_mask, decoder_padding_mask
    #     # )
    #     #
    #     # # Apply the final layer of the transformer to get predictions
    #     # student_prediction = self.transformer.final_layer(student_decoder_output)
    #
    #     # Use the Transformer's call method directly
    #     student_prediction, _ = self.transformer(
    #         src, target_input, True, encoder_padding_mask, combined_mask, decoder_padding_mask
    #     )
    #
    #     return student_prediction
