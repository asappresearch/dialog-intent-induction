"""
cluster metrics: precision, recall, )f1)
"""
import math
import random
from collections import Counter

import scipy.optimize
import numpy as np
import torch


def calc_precision(gnd_assignments, pred_assignments):
    """
    gnd_clusters should be a torch tensor of longs, containing
    the assignment to each cluster

    assumes that cluster assignments are 0-based, and no 'holes'
    """
    precision_sum = 0
    assert len(gnd_assignments.size()) == 1
    assert len(pred_assignments.size()) == 1
    assert pred_assignments.size(0) == gnd_assignments.size(0)
    N = gnd_assignments.size(0)
    K_gnd = gnd_assignments.max().item() + 1
    K_pred = pred_assignments.max().item() + 1
    for k_pred in range(K_pred):
        mask = pred_assignments == k_pred
        gnd = gnd_assignments[mask.nonzero().long().view(-1)]
        max_intersect = 0
        for k_gnd in range(K_gnd):
            intersect = (gnd == k_gnd).long().sum().item()
            max_intersect = max(max_intersect, intersect)
        precision_sum += max_intersect
    precision = precision_sum / N
    return precision


def calc_recall(gnd_assignments, pred_assignments):
    """
    basically the reverse of calc_precision

    so, we can just call calc_precision in reverse :P
    """
    return calc_precision(gnd_assignments=pred_assignments, pred_assignments=gnd_assignments)


def calc_f1(gnd_assignments, pred_assignments):
    prec = calc_precision(gnd_assignments=gnd_assignments, pred_assignments=pred_assignments)
    recall = calc_recall(gnd_assignments=gnd_assignments, pred_assignments=pred_assignments)
    f1 = 2 * (prec * recall) / (prec + recall)
    return f1


def calc_prec_rec_f1(gnd_assignments, pred_assignments):
    prec = calc_precision(gnd_assignments=gnd_assignments, pred_assignments=pred_assignments)
    recall = calc_recall(gnd_assignments=gnd_assignments, pred_assignments=pred_assignments)
    f1 = 2 * (prec * recall) / (prec + recall)
    return prec, recall, f1


def calc_ACC(pred, gnd):
    assert len(pred.size()) == 1
    assert len(gnd.size()) == 1
    N = pred.size(0)
    assert N == gnd.size(0)
    M = torch.zeros(N, N, dtype=torch.int64)
    counts = Counter(list(zip(gnd.tolist(), pred.tolist())))
    keys = torch.LongTensor(list(counts.keys()))
    values = torch.LongTensor(list(counts.values()))
    M = scipy.sparse.csr_matrix((values.numpy(), (keys[:, 0].numpy(), keys[:, 1].numpy())))
    M = M.todense()

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(-M)
    cost = M[row_ind, col_ind].sum().item()
    ACC = cost / N

    return ACC
