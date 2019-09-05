"""
pretrain using qt, to get neural representation, then run kmeans on various
combinations of the resulting representations

this was forked initially from train.py, then modified
"""
import argparse, os, datetime, copy
import time
import numpy as np
np.random.seed(0)
import sklearn.cluster
import warnings
warnings.filterwarnings(action='ignore', category=RuntimeWarning)

LSTM_LAYER = 1
LSTM_HIDDEN = 300
WORD_DROPOUT_RATE = 0.
DROPOUT_RATE = 0.
BATCH_SIZE = 32
LEARNING_RATE = 0.001

import torch
torch.manual_seed(0)
import torch.nn.functional as F
from torch import autograd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from proc_data import Dataset
from model.multiview_encoders import MultiviewEncoders
from metrics import cluster_metrics
from samplers import CategoriesSampler
from model.utils import *
import train
import pretrain


def transform(data, model):
    model.eval()
    latent_zs = []
    n_batch = (len(data) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(n_batch):
        data_batch = data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        with autograd.no_grad():
            latent_z = model(data_batch, encoder='v1')
        latent_zs.append(latent_z.cpu().data.numpy())
    latent_zs = np.concatenate(latent_zs)
    return latent_zs

def calc_prec_rec_f1_acc(preds, golds):
        lgolds, lpreds = [], []
        for g, p in zip(golds, list(preds)):
            if g > 0:
                lgolds.append(g)
                lpreds.append(p)
        prec, rec, f1 = cluster_metrics.calc_prec_rec_f1(gnd_assignments=torch.LongTensor(lgolds).to(device), pred_assignments=torch.LongTensor(lpreds).to(device))
        acc = cluster_metrics.calc_ACC(torch.LongTensor(lpreds).to(device), torch.LongTensor(lgolds).to(device))
        return prec, rec, f1, acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='./data/airlines_processed.csv')
    parser.add_argument('--glove-path', type=str, default='./data/glove.840B.300d.txt')
    parser.add_argument('--pre-epoch', type=int, default=5)
    parser.add_argument('--pt-batch', type=int, default=100)
    parser.add_argument('--scenarios', type=str, default='view1,view2,concatviews,wholeconv', help='comma-separated, from [view1|view2|concatviews|wholeconv|mvsc]')
    parser.add_argument('--mvsc-no-unk', action='store_true', help='only feed non-unk data to MVSC (to avoid oom)')

    parser.add_argument('--view1-col', type=str, default='view1')
    parser.add_argument('--view2-col', type=str, default='view2')
    parser.add_argument('--label-col', type=str, default='tag')
    args = parser.parse_args()

    print('loading dataset')
    dataset = Dataset(args.data_path, view1_col=args.view1_col, view2_col=args.view2_col, label_col=args.label_col)
    n_cluster = len(dataset.id_to_label) - 1
    print ("num of class = %d" %n_cluster)

    id_to_token, token_to_id = dataset.id_to_token, dataset.token_to_id
    pad_idx, start_idx, end_idx = token_to_id["__PAD__"], token_to_id["__START__"], token_to_id["__END__"]
    vocab_size = len(dataset.token_to_id)
    print('vocab_size', vocab_size)

    # Load pre-trained GloVe vectors
    pretrained = {}
    word_emb_size = 0
    print('loading glove')
    for line in open(args.glove_path):
        parts = line.strip().split()
        if len(parts) % 100 != 1: continue
        word = parts[0]
        if word not in token_to_id:
            continue
        vector = [float(v) for v in parts[1:]]
        pretrained[word] = vector
        word_emb_size = len(vector)
    pretrained_list = []
    scale = np.sqrt(3.0 / word_emb_size)
    print('loading oov')
    for word in id_to_token:
        # apply lower() because all GloVe vectors are for lowercase words
        if word.lower() in pretrained:
            pretrained_list.append(np.array(pretrained[word.lower()]))
        else:
            random_vector = np.random.uniform(-scale, scale, [word_emb_size])
            pretrained_list.append(random_vector)

    model = MultiviewEncoders.construct_from_embeddings(
        embeddings=torch.FloatTensor(pretrained_list),
        num_layers=LSTM_LAYER,
        embedding_size=word_emb_size,
        lstm_hidden_size=LSTM_HIDDEN,
        word_dropout=WORD_DROPOUT_RATE,
        dropout=DROPOUT_RATE,
        vocab_size=vocab_size
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    expressions = (model, optimizer)
    pre_acc, pre_state = 0., None
    pretrain_method = pretrain.pretrain_qt
    for epoch in range(1, args.pre_epoch + 1):
        model.train()
        perm_idx = np.random.permutation(dataset.trn_idx)
        trn_loss, _ = pretrain_method(dataset, perm_idx, expressions, train=True)
        model.eval()
        _, tst_acc = pretrain_method(dataset, dataset.tst_idx, expressions, train=False)
        if tst_acc > pre_acc:
            pre_state = copy.deepcopy(model.state_dict())
            pre_acc = tst_acc
        print('{} epoch {}, train_loss={:.4f} test_acc={:.4f}'.format(datetime.datetime.now(), epoch, trn_loss, tst_acc))

    if args.pre_epoch > 0:
        # load best state
        model.load_state_dict(pre_state)

        # deepcopy pretrained views into v1 and/or view2
        pretrain.after_pretrain_qt(model)

    kmeans = sklearn.cluster.KMeans(n_clusters=n_cluster, max_iter=300, verbose=0, random_state=0)

    golds = [dataset[idx][1] for idx in dataset.trn_idx]
    for rep in args.scenarios.split(','):
        if rep == 'view1':
            data = [dataset[idx][0][0] for idx in dataset.trn_idx]
            encoded = transform(data=data, model=model)
            preds = kmeans.fit_predict(encoded)
        elif rep == 'view2':
            data = [dataset[idx][0][1] for idx in dataset.trn_idx]
            encoded = []
            for conv in data:
                encoded_conv = transform(data=conv, model=model)
                encoded_conv = torch.from_numpy(encoded_conv)
                encoded_conv = encoded_conv.mean(dim=0)
                encoded.append(encoded_conv)
            encoded = torch.stack(encoded, dim=0)
            # print('encoded.size()', encoded.size())
            encoded = encoded.numpy()
            preds = kmeans.fit_predict(encoded)
        elif rep == 'concatviews':
            v1_data = [dataset[idx][0][0] for idx in dataset.trn_idx]
            v1_encoded = torch.from_numpy(transform(data=v1_data, model=model))

            v2_data = [dataset[idx][0][1] for idx in dataset.trn_idx]
            v2_encoded = []
            for conv in v2_data:
                encoded_conv = transform(data=conv, model=model)
                encoded_conv = torch.from_numpy(encoded_conv)
                encoded_conv = encoded_conv.mean(dim=0)
                v2_encoded.append(encoded_conv)
            v2_encoded = torch.stack(v2_encoded, dim=0)
            concatview = torch.cat([v1_encoded, v2_encoded], dim=-1)
            print('concatview.size()', concatview.size())
            encoded = concatview.numpy()
            preds = kmeans.fit_predict(encoded)
        elif rep == 'wholeconv':
            encoded = []
            for idx in dataset.trn_idx:
                v1 = dataset[idx][0][0]
                v2 = dataset[idx][0][1]
                conv = [v1] + v2
                encoded_conv = transform(data=conv, model=model)
                encoded_conv = torch.from_numpy(encoded_conv)
                encoded_conv = encoded_conv.mean(dim=0)
                encoded.append(encoded_conv)
            encoded = torch.stack(encoded, dim=0)
            print('encoded.size()', encoded.size())
            encoded = encoded.numpy()
            preds = kmeans.fit_predict(encoded)
        elif rep == 'mvsc':
            try:
                import multiview
            except:
                print('please install https://github.com/mariceli3/multiview')
                return
            print('imported multiview ok')

            idx = dataset.trn_idx_no_unk if args.mvsc_no_unk else dataset.trn_idx
            v1_data = [dataset[idx][0][0] for idx in idx]
            v1_encoded = torch.from_numpy(transform(data=v1_data, model=model))

            v2_data = [dataset[idx][0][1] for idx in idx]
            v2_encoded = []
            for conv in v2_data:
                encoded_conv = transform(data=conv, model=model)
                encoded_conv = torch.from_numpy(encoded_conv)
                encoded_conv = encoded_conv.mean(dim=0)
                v2_encoded.append(encoded_conv)
            v2_encoded = torch.stack(v2_encoded, dim=0)

            mvsc = multiview.mvsc.MVSC(
                k=n_cluster
            )
            print('running mvsc', end='', flush=True)
            start = time.time()
            preds, eivalues, eivectors, sigmas = mvsc.fit_transform(
                [v1_encoded, v2_encoded], [False] * 2
            )
            print('...done')
            mvsc_time = time.time() - start
            print('time taken %.3f' % mvsc_time)
        else:
            raise Exception('unimplemented rep', rep)

        prec, rec, f1, acc = calc_prec_rec_f1_acc(preds, golds)
        print('{} {}: eval prec={:.4f} rec={:.4f} f1={:.4f} acc={:.4f}'.format(datetime.datetime.now(), rep, prec, rec, f1, acc))


if __name__ == '__main__':
    main()
