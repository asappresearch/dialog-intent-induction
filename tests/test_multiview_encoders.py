import torch
import numpy as np

from model.multiview_encoders import MultiviewEncoders


def test_ae_decoder():
    dropout = word_dropout = 0
    embedding_size = 15
    num_layers = 1
    vocab_size = 13
    lstm_hidden_size = 11

    batch_size = 5
    seq_len = 6

    print('vocab_size', vocab_size)
    print('batch_size', batch_size)
    print('seq_len', seq_len)
    print('lstm_hidden_size', lstm_hidden_size)

    _input_idxes = torch.from_numpy(np.random.choice(
        vocab_size - 1, (batch_size, seq_len), replace=True)) + 1
    print('_input_idxes', _input_idxes)
    print('_input_idxes.size()', _input_idxes.size())
    input_idxes = []
    for b in range(batch_size):
        idxes = [_input_idxes[b][i].item() for i in range(seq_len)]
        input_idxes.append(idxes)
    latent_z = torch.rand((batch_size, lstm_hidden_size * 2))
    print('latent_z.size()', latent_z.size())
    print('input_idxes', input_idxes)

    encoders = MultiviewEncoders(vocab_size, num_layers, embedding_size, lstm_hidden_size,
                                 word_dropout, dropout)
    logits = encoders.decode(decoder_input=input_idxes, latent_z=latent_z)
    print('logits.size()', logits.size())


def test_reconst_loss():
    dropout = word_dropout = 0
    embedding_size = 15
    num_layers = 1
    lstm_hidden_size = 11

    vocab_size = 13
    batch_size = 5
    seq_len = 6

    print('vocab_size', vocab_size)
    print('batch_size', batch_size)
    print('seq_len', seq_len)
    print('lstm_hidden_size', lstm_hidden_size)

    _input_idxes = torch.from_numpy(np.random.choice(
        vocab_size - 1, (batch_size, seq_len), replace=True)) + 1
    print('_input_idxes', _input_idxes)
    print('_input_idxes.size()', _input_idxes.size())
    input_idxes = []
    for b in range(batch_size):
        idxes = [_input_idxes[b][i].item() for i in range(seq_len)]
        input_idxes.append(idxes)

    encoders = MultiviewEncoders(vocab_size, num_layers, embedding_size, lstm_hidden_size,
                                 word_dropout, dropout)

    probs = torch.zeros(batch_size, seq_len + 1, vocab_size)
    probs[:, seq_len, encoders.end_idx] = 1
    for b in range(batch_size):
        for i, idx in enumerate(input_idxes[b]):
            probs[b, i, idx] = 1
    logits = probs.log()
    print('logits.sum(dim=-1)', logits.sum(dim=-1))
    print('logits.min(dim=-1)', logits.min(dim=-1)[0])
    print('logits.max(dim=-1)', logits.max(dim=-1)[0])
    _, logits_max = logits.max(dim=-1)
    print('logits_max', logits_max)
    assert (logits_max[:, :seq_len] == _input_idxes).all()

    loss, acc = encoders.reconst_loss(input_idxes, logits)
    loss = loss.item()
    print('loss', loss, 'acc', acc)
    assert acc == 1.0
    assert loss == 0.0
