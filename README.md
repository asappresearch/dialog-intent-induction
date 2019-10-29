Code and data for paper ["Dialog Intent Induction with Deep Multi-View Clustering"](https://arxiv.org/abs/1908.11487), Hugh Perkins and Yi Yang, 2019, to appear in EMNLP 2019.

Data is available in the sub-directory [data](data), with a specific [LICENSE](data/LICENSE) file.

## Concepts

### New Task: Dialog Intent Induction

**The Gap between Academics and Industry**

**Academic dialog** datasets such as ATIS and MultiWoZ assume dialog intents are given; they also focus on simple dialog intents like `BookRestaurant` or `BookHotel`.
Many complex dialog intents emerge in **industrial settings** that are hard to predefine; the dialog intents are also undergoing dynamic changes.

**The New Task: Dialog Intent Induction**

- Automatic discovery of dialog intents from human-human conversations.
- Unsupervised clustering of user query utterances into dialog intents.
- Heuristic: user query utterance is the first user utterance.

### New algorithm: AV-Kmeans

**Our approach: Alternating-View K-Means**

<img src="images/avkmeans_graph.png" width="500" />

We perform clustering on view 1 and project the assignment to view 2 for classification. The encoders are fixed for clustering and updated for classification.

### Experiments

We construct a new dataset to evaluate this new intent induction task:

**Twitter Airlines Customer Support (TwACS)**

We use dialogs in the airline industry from the Kaggle Twitter Support dataset. There are 43,072 unlabeled dialogs. We annotated 500 dialogs to create a test set. We ended up with 14 dialog intents.

We compare three competitive clustering methods: k-means, [Multi-View Spectral Clustering (MVSC)](https://github.com/mariceli3/multiview), `AV-Kmeans` (ours).

We compare three approaches to parameter initialization: PCA for k-means and MVSC; autoencoders; and [quick thoughts](https://arxiv.org/pdf/1803.02893.pdf).

**TwACS Results**

F1 scores:

|Algo   | PCA/None | autoencoders | quick thoughts |
|------|----------|--------------|----------------|
|k-means| 28.2 | 29.5 | 42.1|
|MVSC| 27.8 | 31.3 | 40 |
|**AV-Kmeans (ours)** | **35.4** | **38.9** | **46.2** |

`AV-Kmeans` largely outperforms single-view and classical multi-view clustering methods. Quick thoughts pretraining leads to better results than PCA or autoencoders.

# Usage

## Pre-requisites

- decompress the `.bz2` files in `data`folder
- download http://nlp.stanford.edu/data/glove.840B.300d.zip, and unzip `glove.840B.300d.txt` into `data` folder

## To run AV-Kmeans

- run one of:
```
# no pre-training
python train.py --pre-epoch 0 --data-path data/airlines_processed.csv --num-epochs 50 --view1-col first_utterance --view2-col context

# ae pre-training
python train.py --pre-model ae --pre-epoch 20 --data-path data/airlines_processed.csv --num-epochs 50 --view1-col first_utterance --view2-col context

# qt pre-training
python train.py --pre-model qt --pre-epoch 10 --data-path data/airlines_processed.csv --num-epochs 50 --view1-col first_utterance --view2-col context
```
- to train on askubuntu, replace `airlines` with `askubuntu` in the above command-lines

## To run k-means baseline

- for qt pretraining run:
```
python train_qt.py --data-path  data/airlines_processed.csv --pre-epoch 10 --view1-col first_utterance --view2-col context --scenarios view1
```
- to train on askubuntu, replace `airlines` with `askubuntu` in the above command-line

