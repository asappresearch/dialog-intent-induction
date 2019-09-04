import torch as t
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, latent_z_size, word_emb_size, word_vocab_size, decoder_rnn_size, decoder_num_layers, dropout=0.5):
        super(Decoder, self).__init__()
        self.latent_z_size = latent_z_size
        self.word_vocab_size = word_vocab_size
        self.decoder_rnn_size = decoder_rnn_size
        self.dropout = dropout
        self.rnn = nn.LSTM(input_size=latent_z_size + word_emb_size,
                           hidden_size=decoder_rnn_size,
                           num_layers=decoder_num_layers,
                           dropout=dropout,
                           batch_first=True)
        self.fc = nn.Linear(decoder_rnn_size, word_vocab_size)

    def forward(self, decoder_input, latent_z):
        """
        :param decoder_input: tensor with shape of [batch_size, seq_len, emb_size]
        :param latent_z: sequence context with shape of [batch_size, latent_z_size]
        :return: unnormalized logits of sentense words distribution probabilities
                     with shape of [batch_size, seq_len, word_vocab_size]

        TODO: add padding support
        """

        [batch_size, seq_len, _] = decoder_input.size()
        # decoder rnn is conditioned on context via additional bias = W_cond * z to every input token
        latent_z = t.cat([latent_z] * seq_len, 1).view(batch_size, seq_len, self.latent_z_size)
        decoder_input = t.cat([decoder_input, latent_z], 2)
        rnn_out, _ = self.rnn(decoder_input)
        rnn_out = rnn_out.contiguous().view(-1, self.decoder_rnn_size)
        result = self.fc(rnn_out)
        result = result.view(batch_size, seq_len, self.word_vocab_size)
        return result
