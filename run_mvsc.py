"""
Given a pre-trained model, run mvsc on it, and print scores vs gold standard

we'll use view1 encoder to encode each of view1 and view2, and then pass that through mvsc algo

this should probably be folded innto run_clustering.py (originally kind of forked from run_clustering.py, and combined with some things
from train_pca.py and train.py)
"""
import time, json, random, datetime, argparse, os
from os import path
from os.path import join

try:
    import multiview
except Exception as e:
    print('please install https://github.com/mariceli3/multiview')
    print('eg pip install git+https://github.com/mariceli3/multiview')
    raise e

import sklearn.cluster
import numpy as np, torch
from torch import autograd
from tensorboardX import SummaryWriter

from metrics import cluster_metrics
from model import multiview_encoders
from proc_data import Dataset
import train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 32

def transform(dataset, perm_idx, model, view):
    """
    for view1 utterance, simply encode using view1 encoder
    for view 2 utterances:
    - encode each utterance, using view 1 encoder, to get utterance embeddings
    - take average of utterance embeddings to form view 2 embedding
    """
    model.eval()
    latent_zs, golds = [], []
    n_batch = (len(perm_idx) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(n_batch):
        indices = perm_idx[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        v1_batch, v2_batch = list(zip(*[dataset[idx][0] for idx in indices]))
        golds += [dataset[idx][1] for idx in indices]
        if view == 'v1':
            latent_z = model(v1_batch, encoder='v1')
        elif view == 'v2':
            latent_z_l = [model(conv, encoder='v1').mean(dim=0) for conv in v2_batch]
            latent_z = torch.stack(latent_z_l)
        latent_zs.append(latent_z.cpu().data.numpy())
    latent_zs = np.concatenate(latent_zs)
    return latent_zs, golds

def run(
        ref, model_path, num_clusters, num_cluster_samples, seed,
        out_cluster_samples_file_hier,
        max_examples, out_cluster_samples_file,
        data_path, view1_col, view2_col, label_col,
        sampling_strategy, mvsc_no_unk
    ):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    datetime_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/{ref}/{datetime_str}')

    id_to_token, token_to_id, vocab_size, word_emb_size, mvc_encoder = multiview_encoders.load_model(model_path)
    print('loaded model')

    print('loading dataset')
    dataset = Dataset(data_path, view1_col=view1_col, view2_col=view2_col, label_col=label_col)
    n_cluster = len(dataset.id_to_label) - 1
    print ("loaded dataset, num of class = %d" %n_cluster)

    idxes = dataset.trn_idx_no_unk if mvsc_no_unk else dataset.trn_idx
    trn_idx = [x.item() for x in np.random.permutation(idxes)]
    if max_examples is not None:
        trn_idx = trn_idx[:max_examples]

    num_clusters = n_cluster if num_clusters is None else num_clusters
    print('clustering over num clusters', num_clusters)

    mvsc = multiview.mvsc.MVSC(
        k=n_cluster
    )
    latent_z1s, golds = transform(dataset, trn_idx, mvc_encoder, view='v1')
    latent_z2s, _ = transform(dataset, trn_idx, mvc_encoder, view='v2')
    print('running mvsc', end='', flush=True)
    start = time.time()
    preds, eivalues, eivectors, sigmas = mvsc.fit_transform(
        [latent_z1s, latent_z2s], [False] * 2
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
    silhouette, davies_bouldin = sklearn.metrics.silhouette_score(latent_z1s, preds, metric='euclidean'), sklearn.metrics.davies_bouldin_score(latent_z1s, preds)
    print('{} pretrain: eval prec={:.4f} rec={:.4f} f1={:.4f} acc={:.4f} sil={:.4f}, db={:.4f}'.format(datetime.datetime.now(), prec, rec, f1, acc, silhouette, davies_bouldin))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--max-examples', type=int, help='since we might not want to cluster entire dataset?')
    parser.add_argument('--mvsc-no-unk', action='store_true', help='only feed non-unk data to MVSC (to avoid oom)')
    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--data-path', type=str, default='./data/airlines_500_merged.csv')
    parser.add_argument('--view1-col', type=str, default='view1_col')
    parser.add_argument('--view2-col', type=str, default='view2_col')
    parser.add_argument('--label-col', type=str, default='cluster_id')
    parser.add_argument('--num-clusters', type=int, help='defaults to number of supervised classes')
    parser.add_argument('--num-cluster-samples', type=int, default=10)
    parser.add_argument('--sampling-strategy', type=str, choices=['uniform', 'nearest'], default='nearest')
    parser.add_argument('--out-cluster-samples-file-hier', type=str, default='tmp/cluster_samples_hier_{ref}.txt')
    parser.add_argument('--out-cluster-samples-file', type=str, default='tmp/cluster_samples_{ref}.txt')
    args = parser.parse_args()
    args.out_cluster_samples_file = args.out_cluster_samples_file.format(**args.__dict__)
    args.out_cluster_samples_file_hier = args.out_cluster_samples_file_hier.format(**args.__dict__)
    run(**args.__dict__)
