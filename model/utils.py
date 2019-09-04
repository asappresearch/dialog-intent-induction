import os, time, pprint
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pad_sentences(sents, lpad=None, rpad=None, reverse=False, pad_idx=0, max_sent_len=100):
    sentences = []
    max_len = 0
    for i in range(len(sents)):
        if len(sents[i]) > max_sent_len:
            sentences.append(sents[i][:max_sent_len])
        else:
            sentences.append(sents[i])
        max_len = max(max_len, len(sentences[i]))
    if reverse:
        sentences = [sentence[::-1] for sentence in sentences]
    if lpad is not None:
        sentences = [[lpad] + sentence for sentence in sentences]
        max_len += 1
    if rpad is not None:
        sentences = [sentence + [rpad] for sentence in sentences]
        max_len += 1
    lengths = []
    for i in range(len(sentences)):
        lengths.append(len(sentences[i]))
        sentences[i] = sentences[i] + [pad_idx]*(max_len - len(sentences[i]))
    return torch.LongTensor(sentences).to(device), torch.LongTensor(lengths).to(device)

def pad_paragraphs(paras, pad_idx=0):
    sentences, lengths = [], []
    max_len = 0
    for para in paras: max_len = max(max_len, len(para))
    for para in paras:
        for sent in para:
            sentences.append(sent[:])
        lengths.append(len(para))
        for i in range(max_len - len(para)):
            sentences.append([pad_idx])
    ret_sents, sent_lens = pad_sentences(sentences, pad_idx=pad_idx)
    return ret_sents, sent_lens, torch.LongTensor(lengths).to(device), max_len

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits


