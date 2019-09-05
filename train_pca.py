"""
train PCA baseline

forked from train.py, then MVC stripped, and replaced with BoW + PCA

if using mvsc, needs installation of:
- https://github.com/mariceli3/multiview
"""
import argparse, os, datetime, copy
import numpy as np
np.random.seed(0)
import sklearn.cluster
import warnings
import time
warnings.filterwarnings(action='ignore', category=RuntimeWarning)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD

import torch
torch.manual_seed(0)
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from proc_data import Dataset
from metrics import cluster_metrics
from model.utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='./data/airlines_500_mergedb.csv')
    parser.add_argument('--model', type=str, choices=['view1pca', 'view2pca', 'wholeconvpca', 'mvsc'], default='view1pca')
    parser.add_argument('--pca-dims', type=int, default=600)

    parser.add_argument('--no-idf', action='store_true')
    parser.add_argument('--mvsc-no-unk', action='store_true', help='only feed non-unk data to MVSC (to avoid oom)')

    parser.add_argument('--view1-col', type=str, default='view1')
    parser.add_argument('--view2-col', type=str, default='view2')
    parser.add_argument('--label-col', type=str, default='tag')
    args = parser.parse_args()
    run(**args.__dict__)


def run(data_path, model, pca_dims, view1_col, view2_col, label_col, no_idf, mvsc_no_unk):
    print('loading dataset')
    dataset = Dataset(data_path, view1_col=view1_col, view2_col=view2_col, label_col=label_col)
    n_cluster = len(dataset.id_to_label) - 1
    print ("num of class = %d" %n_cluster)

    id_to_token, token_to_id = dataset.id_to_token, dataset.token_to_id
    pad_idx, start_idx, end_idx = token_to_id["__PAD__"], token_to_id["__START__"], token_to_id["__END__"]
    vocab_size = len(dataset.token_to_id)
    print('vocab_size', vocab_size)

    if model == 'mvsc':
        try:
            import multiview
        except:
            print('please install https://github.com/mariceli3/multiview')
            return
        print('imported multiview ok')

    def run_pca(features):
        print('fitting tfidf vectorizer', flush=True, end='')
        vectorizer = TfidfVectorizer(token_pattern='\d+', ngram_range=(1, 1), analyzer='word', min_df=0.0, max_df=1.0, use_idf=not no_idf)
        X = vectorizer.fit_transform(features)
        print(' ... done')
        print('X.shape', X.shape)

        print('running pca', flush=True, end='')
        pca = TruncatedSVD(n_components=pca_dims)
        X2 = pca.fit_transform(X)
        print(' ... done')
        return X2

    golds = [dataset[idx][1] for idx in dataset.trn_idx]

    if model in ['view1pca', 'view2pca', 'wholeconvpca']:
        if model == 'view1pca':
            utts = [dataset[idx][0][0] for idx in dataset.trn_idx]
            utts = [' '.join([str(idx) for idx in utt]) for utt in utts]
        elif model == 'view2pca':
            convs = [dataset[idx][0][1] for idx in dataset.trn_idx]
            utts = [[tok for utt in conv for tok in utt] for conv in convs]
            utts = [' '.join([str(idx) for idx in utt]) for utt in utts]
        elif model == 'wholeconvpca':
            v1 = [dataset[idx][0][0] for idx in dataset.trn_idx]
            convs = [dataset[idx][0][1] for idx in dataset.trn_idx]
            v2 = [[tok for utt in conv for tok in utt] for conv in convs]
            utts = []
            for n in range(len(v1)):
                utts.append(v1[n] + v2[n])
            utts = [' '.join([str(idx) for idx in utt]) for utt in utts]

        X2 = run_pca(utts)

        print('running kmeans', flush=True, end='')
        kmeans = sklearn.cluster.KMeans(n_clusters=n_cluster, max_iter=300, verbose=0, random_state=0)
        preds = kmeans.fit_predict(X2)
        print(' ... done')

    elif model == 'mvsc':
        mvsc = multiview.mvsc.MVSC(
            k=n_cluster
        )
        idxes = dataset.trn_idx_no_unk if mvsc_no_unk else dataset.trn_idx
        v1 = [dataset[idx][0][0] for idx in idxes]
        convs = [dataset[idx][0][1] for idx in idxes]
        v2 = [[tok for utt in conv for tok in utt] for conv in convs]
        v1 = [' '.join([str(idx) for idx in utt]) for utt in v1]
        v2 = [' '.join([str(idx) for idx in utt]) for utt in v2]
        v1_pca = run_pca(v1)
        v2_pca = run_pca(v2)
        print('running mvsc', end='', flush=True)
        start = time.time()
        preds, eivalues, eivectors, sigmas = mvsc.fit_transform(
            [v1_pca, v2_pca], [False] * 2
        )
        print('...done')
        mvsc_time = time.time() - start
        print('time taken %.3f' % mvsc_time)

    lgolds, lpreds = [], []
    for g, p in zip(golds, list(preds)):
        if g > 0:
            lgolds.append(g)
            lpreds.append(p)
    prec, rec, f1 = cluster_metrics.calc_prec_rec_f1(gnd_assignments=torch.LongTensor(lgolds).to(device), pred_assignments=torch.LongTensor(lpreds).to(device))
    acc = cluster_metrics.calc_ACC(torch.LongTensor(lpreds).to(device), torch.LongTensor(lgolds).to(device))

    print('{} eval f1={:.4f} prec={:.4f} rec={:.4f} acc={:.4f}'.format(datetime.datetime.now(), prec, rec, f1, acc))

    return prec, rec, f1, acc


if __name__ == '__main__':
    main()
