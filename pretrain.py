import copy
import torch
import numpy as np

from train import BATCH_SIZE, AE_BATCH_SIZE

def pretrain_qt(dataset, perm_idx, expressions, train=True):
    """
    for each pair of utterances:
    - encodes first utterance using 'v1' encoder
    - encodes second utterance using 'qt_context' encoder
    uses negative sampling loss between these two embeddings, relative to the
    other second utterances in the batch
    """
    model, optimizer = expressions

    utts = []
    qt_ex = []
    for idx in perm_idx:
        v1, v2 = dataset[idx][0]
        conversation = [v1] + v2
        for n, utt in enumerate(conversation):
            utts.append(utt)
            if n > 0:
                num_utt = len(utts)
                ex = (num_utt - 2, num_utt - 1)
                qt_ex.append(ex)
    qt_ex = np.random.permutation(qt_ex)

    total_loss, total_acc = 0., 0.
    n_batch = (len(qt_ex) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(n_batch):
        qt_ex_batch = qt_ex[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

        v1_idxes, v2_idxes = list(zip(*[(ex[0].item(), ex[1].item()) for ex in qt_ex_batch]))
        v1_utts = [utts[idx] for idx in v1_idxes]
        v2_utts = [utts[idx] for idx in v2_idxes]

        v1_state = model(v1_utts, encoder='v1')
        v2_state = model(v2_utts, encoder='qt')

        loss, acc = model.qt_loss(v2_state, v1_state)
        total_loss += loss.item()
        total_acc += acc * len(qt_ex_batch)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return total_loss, total_acc / len(qt_ex)

def after_pretrain_qt(model):
    model.view2_word_rnn = copy.deepcopy(model.view1_word_rnn)

def pretrain_ae(dataset, perm_idx, expressions, train=True):
    """
    uses v1 encoder to encode all utterances in both view1 and view2
    to utterance-level embeddings
    uses 'ae_decoder' rnn from model to decode these embeddings
    (works at utterance level)
    """
    model, optimizer = expressions

    utterances = []
    for idx in perm_idx:
        v1, v2 = dataset[idx][0]
        conversation = [v1] + v2
        utterances += conversation
    utterances = np.random.permutation(utterances)

    total_loss, total_acc = 0., 0.
    n_batch = (len(utterances) + AE_BATCH_SIZE - 1) // AE_BATCH_SIZE
    for i in range(n_batch):
        utt_batch = utterances[i*AE_BATCH_SIZE:(i+1)*AE_BATCH_SIZE]
        enc_state = model(utt_batch, encoder='v1')
        reconst = model.decode(decoder_input=utt_batch, latent_z=enc_state)
        loss, acc = model.reconst_loss(utt_batch, reconst)

        total_loss += loss.item()
        total_acc += acc * len(utt_batch)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    total_acc = total_acc / len(utterances)
    return total_loss, total_acc

def after_pretrain_ae(model):
    # we'll use the view1 encoder for both view 1 and view 2
    model.view2_word_rnn = copy.deepcopy(model.view1_word_rnn)
