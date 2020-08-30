import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, word_emb_size, encoder_rnn_size, encoder_num_layers, dropout=0.5):
        super(Encoder, self).__init__()

        self.encoder_rnn_size = encoder_rnn_size
        self.encoder_num_layers = encoder_num_layers
        # Create input dropout parameter
        self.rnn = nn.LSTM(input_size=word_emb_size,
                           hidden_size=encoder_rnn_size,
                           num_layers=encoder_num_layers,
                           dropout=dropout,
                           batch_first=True,
                           bidirectional=True)

    def forward(self, encoder_input, lengths):
        """
        :param encoder_input: [batch_size, seq_len, emb_size] tensor
        :return: context of input sentenses with shape of [batch_size, encoder_rnn_size]
        """
        lengths, perm_idx = lengths.sort(0, descending=True)
        encoder_input = encoder_input[perm_idx]
        [batch_size, seq_len, _] = encoder_input.size()
        packed_words = torch.nn.utils.rnn.pack_padded_sequence(
                encoder_input, lengths, True)
        # Unfold rnn with zero initial state and get its final state from the last layer
        rnn_out, (_, final_state) = self.rnn(packed_words, None)
        final_state = final_state.view(
            self.encoder_num_layers, 2, batch_size, self.encoder_rnn_size)[-1]
        h_1, h_2 = final_state[0], final_state[1]
        final_state = torch.cat([h_1, h_2], 1)
        _, unperm_idx = perm_idx.sort(0)
        final_state = final_state[unperm_idx]
        return final_state
