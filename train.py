import argparse
import datetime
import copy
import numpy as np
from os.path import expanduser as expand
import sklearn.cluster
import sklearn.metrics
import warnings

import torch
import torch.nn.functional as F

from proc_data import Dataset
from model import multiview_encoders
from metrics import cluster_metrics
from samplers import CategoriesSampler
from model import utils
import pretrain


warnings.filterwarnings(action='ignore', category=RuntimeWarning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

LSTM_LAYER = 1
LSTM_HIDDEN = 300
WORD_DROPOUT_RATE = 0.
DROPOUT_RATE = 0.
BATCH_SIZE = 32
AE_BATCH_SIZE = 8
LEARNING_RATE = 0.001


def do_pass(batches, shot, way, query, expressions, encoder):
    model, optimizer = expressions
    model.train()
    for i, batch in enumerate(batches, 1):
        data = [x[{'v1': 0, 'v2': 1}[encoder]] for x, _ in batch]
        p = shot * way
        data_shot, data_query = data[:p], data[p:]

        proto = model(data_shot, encoder=encoder)
        proto = proto.reshape(shot, way, -1).mean(dim=0)

        # ignore original labels, reassign labels from 0
        label = torch.arange(way).repeat(query)
        label = label.type(torch.LongTensor).to(device)

        logits = utils.euclidean_metric(model(data_query, encoder=encoder), proto)
        loss = F.cross_entropy(logits, label)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
    return loss.item()


def transform(dataset, perm_idx, model, encoder):
    model.eval()
    latent_zs, golds = [], []
    n_batch = (len(perm_idx) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(n_batch):
        indices = perm_idx[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        v1_batch, v2_batch = list(zip(*[dataset[idx][0] for idx in indices]))
        golds += [dataset[idx][1] for idx in indices]
        latent_z = model({'v1': v1_batch, 'v2': v2_batch}[encoder], encoder=encoder)
        latent_zs.append(latent_z.cpu().data.numpy())
    latent_zs = np.concatenate(latent_zs)
    return latent_zs, golds


def calc_centroids(latent_zs, assignments, n_cluster):
    centroids = []
    for i in range(n_cluster):
        idx = np.where(assignments == i)[0]
        mean = np.mean(latent_zs[idx], 0)
        centroids.append(mean)
    return np.stack(centroids)


def run_one_side(model, optimizer, preds_left, pt_batch, way, shot, query, n_cluster, dataset,
                 right_encoder_side):
    """
    encoder_side should be 'v1' or 'v2'. It should match whichever view is 'right' here.
    """
    loss = 0
    perm_idx = np.random.permutatation(dataset.trn_idx)

    sampler = CategoriesSampler(preds_left, pt_batch, way, shot + query)
    train_batches = [[dataset[perm_idx[idx]] for idx in indices] for indices in sampler]
    loss += do_pass(train_batches, shot, way, query, (model, optimizer), encoder=right_encoder_side)

    z_right, _ = transform(dataset, dataset.trn_idx, model, encoder=right_encoder_side)
    centroids = calc_centroids(z_right, preds_left, n_cluster)
    kmeans = sklearn.cluster.KMeans(
        n_clusters=n_cluster, init=centroids, max_iter=10, verbose=0)
    preds_right = kmeans.fit_predict(z_right)

    tst_z_right, _ = transform(dataset, dataset.tst_idx, model, encoder=right_encoder_side)
    tst_preds_right = kmeans.predict(tst_z_right)

    return loss, preds_right, tst_preds_right


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='./data/airlines_processed.csv')
    parser.add_argument('--glove-path', type=str, default='./data/glove.840B.300d.txt')
    parser.add_argument('--pre-model', type=str, choices=['ae', 'qt'], default='qt')
    parser.add_argument('--pre-epoch', type=int, default=0)
    parser.add_argument('--pt-batch', type=int, default=100)
    parser.add_argument('--model-path', type=str, help='path of pretrained model to load')
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--save-model-path', type=str)

    parser.add_argument('--view1-col', type=str, default='view1')
    parser.add_argument('--view2-col', type=str, default='view2')
    parser.add_argument('--label-col', type=str, default='label')
    args = parser.parse_args()

    np.random.seed(args.seed)

    print('loading dataset')
    dataset = Dataset(
        args.data_path, view1_col=args.view1_col, view2_col=args.view2_col,
        label_col=args.label_col)
    n_cluster = len(dataset.id_to_label) - 1
    print("num of class = %d" % n_cluster)

    if args.model_path is not None:
        id_to_token, token_to_id, vocab_size, word_emb_size, model = multiview_encoders.load_model(
            args.model_path)
        print('loaded model')
    else:
        id_to_token, token_to_id, vocab_size, word_emb_size, model = \
            multiview_encoders.from_embeddings(
                args.glove_path, dataset.id_to_token, dataset.token_to_id)
        print('created randomly initialized model')
    print('vocab_size', vocab_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    expressions = (model, optimizer)

    pre_acc, pre_state, pre_state_epoch = 0., None, None
    pretrain_method = {
        'ae': pretrain.pretrain_ae,
        'qt': pretrain.pretrain_qt,
    }[args.pre_model]
    for epoch in range(1, args.pre_epoch + 1):
        model.train()
        perm_idx = np.random.permutation(dataset.trn_idx)
        trn_loss, _ = pretrain_method(dataset, perm_idx, expressions, train=True)
        model.eval()
        _, tst_acc = pretrain_method(dataset, dataset.tst_idx, expressions, train=False)
        if tst_acc > pre_acc:
            pre_state = copy.deepcopy(model.state_dict())
            pre_acc = tst_acc
            pre_state_epoch = epoch
        print('{} epoch {}, train_loss={:.4f} test_acc={:.4f}'.format(
            datetime.datetime.now(), epoch, trn_loss, tst_acc))

    if args.pre_epoch > 0:
        # load best state
        model.load_state_dict(pre_state)
        print(f'loaded best state from epoch {pre_state_epoch}')

        # deepcopy pretrained views into v1 and/or view2
        {
            'ae': pretrain.after_pretrain_ae,
            'qt': pretrain.after_pretrain_qt,
        }[args.pre_model](model)

        # reinitialiate optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        expressions = (model, optimizer)
        print('applied post-pretraining')

    kmeans = sklearn.cluster.KMeans(n_clusters=n_cluster, max_iter=300, verbose=0, random_state=0)
    z_v1, golds = transform(dataset, dataset.trn_idx, model, encoder='v1')
    preds_v1 = kmeans.fit_predict(z_v1)

    lgolds, lpreds = [], []
    for g, p in zip(golds, list(preds_v1)):
        if g > 0:
            lgolds.append(g)
            lpreds.append(p)
    prec, rec, f1 = cluster_metrics.calc_prec_rec_f1(
        gnd_assignments=torch.LongTensor(lgolds).to(device),
        pred_assignments=torch.LongTensor(lpreds).to(device))
    acc = cluster_metrics.calc_ACC(
        torch.LongTensor(lpreds).to(device), torch.LongTensor(lgolds).to(device))

    print(f'{datetime.datetime.now()} pretrain: test prec={prec:.4f} rec={rec:.4f} '
          f'f1={f1:.4f} acc={acc:.4f}')

    shot, way, query = 5, args.way, 15

    preds_v2 = None
    for epoch in range(1, args.num_epochs + 1):
        trn_loss = 0.

        _loss, preds_v2, tst_preds_v2 = run_one_side(
            model=model, optimizer=optimizer, preds_left=preds_v1,
            pt_batch=args.pt_batch, way=way, shot=shot, query=query, n_cluster=n_cluster,
            dataset=dataset, right_encoder_side='v2')
        trn_loss += _loss

        _loss, preds_v1, tst_preds_v1 = run_one_side(
            model=model, optimizer=optimizer, preds_left=preds_v2,
            pt_batch=args.pt_batch, way=way, shot=shot, query=query, n_cluster=n_cluster,
            dataset=dataset, right_encoder_side='v1')
        trn_loss += _loss

        f1 = cluster_metrics.calc_f1(gnd_assignments=torch.LongTensor(tst_preds_v1).to(device),
                                     pred_assignments=torch.LongTensor(tst_preds_v2).to(device))
        acc = cluster_metrics.calc_ACC(
            torch.LongTensor(tst_preds_v2).to(device), torch.LongTensor(tst_preds_v1).to(device))

        print('eval view 1 vs view 2: f1={:.4f} acc={:.4f}'.format(f1, acc))

        lgolds, lpreds = [], []
        for g, p in zip(golds, list(preds_v1)):
            if g > 0:
                lgolds.append(g)
                lpreds.append(p)
        prec, rec, f1 = cluster_metrics.calc_prec_rec_f1(
            gnd_assignments=torch.LongTensor(lgolds).to(device),
            pred_assignments=torch.LongTensor(lpreds).to(device))
        acc = cluster_metrics.calc_ACC(
            torch.LongTensor(lpreds).to(device), torch.LongTensor(lgolds).to(device))

        print(f'{datetime.datetime.now()} epoch {epoch}, test prec={prec:.4f} rec={rec:.4f} '
              f'f1={f1:.4f} acc={acc:.4f}')

    if args.save_model_path is not None:
        preds_v1 = torch.from_numpy(preds_v1)
        if preds_v2 is not None:
            preds_v2 = torch.from_numpy(preds_v2)
        state = {
            'model_state': model.state_dict(),
            'id_to_token': dataset.id_to_token,
            'word_emb_size': word_emb_size,
            'v1_assignments': preds_v1,
            'v2_assignments': preds_v2
        }
        with open(expand(args.save_model_path), 'wb') as f:
            torch.save(state, f)
        print('saved model to ', args.save_model_path)


if __name__ == '__main__':
    main()
