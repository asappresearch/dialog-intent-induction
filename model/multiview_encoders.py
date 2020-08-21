import torch
import torch as t
from torch import nn
import numpy as np

from model.utils import pad_sentences, pad_paragraphs
import train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MultiviewEncoders(nn.Module):

    def __init__(
            self, vocab_size, num_layers, embedding_size, lstm_hidden_size, word_dropout, dropout,
            start_idx=2, end_idx=3, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.start_idx = start_idx   # for RNN autoencoder training
        self.end_idx = end_idx       # for RNN autoencoder training
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.lstm_hidden_size = lstm_hidden_size
        self.word_dropout = nn.Dropout(word_dropout)
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.crit = nn.CrossEntropyLoss()

        self.embedder = nn.Embedding(vocab_size, embedding_size)

        def create_rnn(embedding_size, bidirectional=True):
            return nn.LSTM(
                embedding_size,
                lstm_hidden_size,
                dropout=dropout,
                num_layers=num_layers,
                bidirectional=bidirectional
            )

        self.view1_word_rnn = create_rnn(embedding_size)
        self.view2_word_rnn = create_rnn(embedding_size)
        self.view2_sent_rnn = create_rnn(2*lstm_hidden_size)

        self.ae_decoder = create_rnn(embedding_size + 2 * lstm_hidden_size, bidirectional=False)
        self.qt_context = create_rnn(embedding_size)

        self.fc = nn.Linear(lstm_hidden_size, vocab_size)

    def get_encoder(self, encoder):
        return {
            'v1': self.view1_word_rnn,
            'v2': self.view2_word_rnn,
            'v2sent': self.view2_sent_rnn,
            'ae_decoder': self.ae_decoder,
            'qt': self.qt_context
        }[encoder]

    @classmethod
    def construct_from_embeddings(
            cls, embeddings, num_layers, embedding_size, lstm_hidden_size, word_dropout, dropout,
            vocab_size, start_idx=2, end_idx=3, pad_idx=0):
        model = cls(
            num_layers=num_layers,
            embedding_size=embedding_size,
            lstm_hidden_size=lstm_hidden_size,
            word_dropout=word_dropout,
            dropout=dropout,
            start_idx=start_idx,
            end_idx=end_idx,
            pad_idx=pad_idx,
            vocab_size=vocab_size
        )
        model.embedder = nn.Embedding.from_pretrained(embeddings, freeze=False)
        return model

    def decode(self, decoder_input, latent_z):
        """
        decode state into word indices

        :param decoder_input: list of lists of indices
        :param latent_z: sequence context with shape of [batch_size, latent_z_size]

        :return: unnormalized logits of sentense words distribution probabilities
                     with shape of [batch_size, seq_len, word_vocab_size]

        """
        padded, lengths = pad_sentences(decoder_input, pad_idx=self.pad_idx, lpad=self.start_idx)
        embeddings = self.embedder(padded)
        embeddings = self.word_dropout(embeddings)
        [batch_size, seq_len, _] = embeddings.size()
        # decoder rnn is conditioned on context via additional bias = W_cond * z
        # to every input token
        latent_z = t.cat([latent_z] * seq_len, 1).view(batch_size, seq_len, -1)
        embeddings = t.cat([embeddings, latent_z], 2)
        rnn = self.ae_decoder
        rnn_out, _ = rnn(embeddings)
        rnn_out = rnn_out.contiguous().view(batch_size * seq_len, self.lstm_hidden_size)
        result = self.fc(rnn_out)
        result = result.view(batch_size, seq_len, self.vocab_size)
        return result

    def forward(self, input, encoder):
        """
        Encode an input into a vector representation
        params:
            input : word indices
            encoder: [pt1|pt2|v1|v2]
        """
        if encoder == 'v2':
            return self.hierarchical_forward(input)

        batch_size = len(input)
        padded, lengths = pad_sentences(input, pad_idx=self.pad_idx)
        embeddings = self.embedder(padded)
        embeddings = self.word_dropout(embeddings)
        lengths, perm_idx = lengths.sort(0, descending=True)
        embeddings = embeddings[perm_idx]
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, True)
        rnn = self.get_encoder(encoder)
        _, (_, final_state) = rnn(packed, None)
        _, unperm_idx = perm_idx.sort(0)
        final_state = final_state[:, unperm_idx]
        final_state = final_state.view(self.num_layers, 2, batch_size, self.lstm_hidden_size)[-1] \
            .transpose(0, 1).contiguous() \
            .view(batch_size, 2 * self.lstm_hidden_size)
        return final_state

    def hierarchical_forward(self, input):
        batch_size = len(input)
        padded, word_lens, sent_lens, max_sent_len = pad_paragraphs(input, pad_idx=self.pad_idx)
        embeddings = self.embedder(padded)
        embeddings = self.word_dropout(embeddings)
        word_lens, perm_idx = word_lens.sort(0, descending=True)
        embeddings = embeddings[perm_idx]
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embeddings, word_lens, True)
        _, (_, final_word_state) = self.view2_word_rnn(packed, None)
        _, unperm_idx = perm_idx.sort(0)
        final_word_state = final_word_state[:, unperm_idx]
        final_word_state = final_word_state.view(
            self.num_layers, 2, batch_size*max_sent_len, self.lstm_hidden_size)[-1] \
            .transpose(0, 1).contiguous() \
            .view(batch_size, max_sent_len, 2 * self.lstm_hidden_size)

        sent_lens, sent_perm_idx = sent_lens.sort(0, descending=True)
        sent_embeddings = final_word_state[sent_perm_idx]
        sent_packed = torch.nn.utils.rnn.pack_padded_sequence(sent_embeddings, sent_lens, True)
        _, (_, final_sent_state) = self.view2_sent_rnn(sent_packed, None)
        _, sent_unperm_idx = sent_perm_idx.sort(0)
        final_sent_state = final_sent_state[:, sent_unperm_idx]
        final_sent_state = final_sent_state.view(
            self.num_layers, 2, batch_size, self.lstm_hidden_size)[-1] \
            .transpose(0, 1).contiguous() \
            .view(batch_size, 2 * self.lstm_hidden_size)
        return final_sent_state

    def qt_loss(self, target_view_state, input_view_state):
        """
        pick out the correct example in the target_view, based on the corresponding input_view
        """
        scores = input_view_state @ target_view_state.transpose(0, 1)
        batch_size = scores.size(0)
        targets = torch.from_numpy(np.arange(batch_size, dtype=np.int64))
        targets = targets.to(scores.device)
        loss = self.crit(scores, targets)
        _, argmax = scores.max(dim=-1)
        examples_correct = (argmax == targets)
        acc = examples_correct.float().mean().item()
        return loss, acc

    def reconst_loss(self, gnd_utts, reconst):
        """
        gnd_utts is a list of lists of indices (the outer list should be a minibatch)
        reconst is a tensor with the logits from a decoder [batchsize][seqlen][vocabsize]
        """
        batch_size, seq_len, vocab_size = reconst.size()
        loss = 0
        padded, lengths = pad_sentences(gnd_utts, pad_idx=self.pad_idx, rpad=self.end_idx)
        batch_size = len(lengths)
        crit = nn.CrossEntropyLoss()
        loss += crit(
            reconst.view(batch_size * seq_len, vocab_size), padded.view(batch_size * seq_len))
        _, argmax = reconst.max(dim=-1)
        correct = (argmax == padded)
        acc = correct.float().mean().item()
        return loss, acc


def from_embeddings(glove_path, id_to_token, token_to_id):
    vocab_size = len(token_to_id)

    # Load pre-trained GloVe vectors
    pretrained = {}
    word_emb_size = 0
    print('loading glove')
    for line in open(glove_path):
        parts = line.strip().split()
        if len(parts) % 100 != 1:
            continue
        word = parts[0]
        if word not in token_to_id:
            continue
        vector = [float(v) for v in parts[1:]]
        pretrained[word] = vector
        word_emb_size = len(vector)
    pretrained_list = []
    scale = np.sqrt(3.0 / word_emb_size)
    print('loading oov')
    for word in token_to_id:
        # apply lower() because all GloVe vectors are for lowercase words
        if word.lower() in pretrained:
            pretrained_list.append(np.array(pretrained[word.lower()]))
        else:
            random_vector = np.random.uniform(-scale, scale, [word_emb_size])
            pretrained_list.append(random_vector)

    print('instantiating model')
    model = MultiviewEncoders.construct_from_embeddings(
        embeddings=torch.FloatTensor(pretrained_list),
        num_layers=train.LSTM_LAYER,
        embedding_size=word_emb_size,
        lstm_hidden_size=train.LSTM_HIDDEN,
        word_dropout=train.WORD_DROPOUT_RATE,
        dropout=train.DROPOUT_RATE,
        vocab_size=vocab_size
    )
    model.to(device)
    return id_to_token, token_to_id, vocab_size, word_emb_size, model


def load_model(model_path):
    with open(model_path, 'rb') as f:
        state = torch.load(f)

    id_to_token = state['id_to_token']
    word_emb_size = state['word_emb_size']

    token_to_id = {token: id for id, token in enumerate(id_to_token)}
    vocab_size = len(id_to_token)

    mvc_encoder = MultiviewEncoders(
        num_layers=train.LSTM_LAYER,
        embedding_size=word_emb_size,
        lstm_hidden_size=train.LSTM_HIDDEN,
        word_dropout=train.WORD_DROPOUT_RATE,
        dropout=train.DROPOUT_RATE,
        vocab_size=vocab_size
    )
    mvc_encoder.to(device)
    mvc_encoder.load_state_dict(state['model_state'])
    return id_to_token, token_to_id, vocab_size, word_emb_size, mvc_encoder
