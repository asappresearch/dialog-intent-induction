Code and data for paper ["Dialog Intent Induction with Deep Multi-View Clustering"](https://arxiv.org/abs/1908.11487), Hugh Perkins and Yi Yang, 2019, to appear in EMNLP 2019.

Data is available in the sub-directory [data](data), with a specific [LICENSE](data/LICENSE) file.

# Dialog Intent Induction

Dialog intent induction aims at automatically discovering dialog intents from human-human conversations. The problem is largely overlooked in the prior academic works, which created a hugh gap between academics and industry. 

In particular, **academic dialog** datasets such as ATIS and MultiWoZ assume dialog intents are given; they also focus on simple dialog intents like `BookRestaurant` or `BookHotel`. However, many complex dialog intents emerge in **industrial settings** that are hard to predefine; the dialog intents are also undergoing dynamic changes.

## Deep multi-view clustering

In this work, we propose to tackle this problem using multi-view clustering. Consider the following example dialogs:

<img src="images/example_dialogs.png" width="350" />

The user query utterances (query view) are lexically and syntactically dissimilar. However, the solution trajectories (content view) are similar. 

### Alternating-view k-means (AV-KMeans)

We propose a novel method for joint representation learning and multi-view cluster: alternating-view k-means (AV-KMeans).

<img src="images/avkmeans_graph.png" width="500" />

We perform clustering on view 1 and project the assignment to view 2 for classification. The encoders are fixed for clustering and updated for classification.

## Experiments

We construct a new dataset to evaluate this new intent induction task: Twitter Airlines Customer Support (TwACS).

We compare three competitive clustering methods: k-means, [Multi-View Spectral Clustering (MVSC)](https://github.com/mariceli3/multiview), and `AV-Kmeans` (ours). We experiment with three approaches to parameter initialization: PCA for k-means and MVSC; autoencoders; and [quick thoughts](https://arxiv.org/pdf/1803.02893.pdf).

The F1 scores are presented below:

|Algo   | PCA/None | autoencoders | quick thoughts |
|------|----------|--------------|----------------|
|k-means| 28.2 | 29.5 | 42.1|
|MVSC| 27.8 | 31.3 | 40 |
|**AV-Kmeans (ours)** | **35.4** | **38.9** | **46.2** |


# Usage

## Pre-requisites

- decompress the `.bz2` files in `data`folder
- download http://nlp.stanford.edu/data/glove.840B.300d.zip, and unzip `glove.840B.300d.txt` into `data` folder

## To run AV-Kmeans

- run one of:
```
# no pre-training
cmdlines/airlines_mvc.sh

# ae pre-training
cmdlines/airlines_ae.sh
cmdlines/airlines_ae_mvc.sh

# qt pre-training
cmdlines/airlines_qt.sh
cmdlines/airlines_qt_mvc.sh
```
- to train on askubuntu, replace `airlines` with `ubuntu` in the above command-lines

## To run k-means baseline

- for qt pretraining run:
```
PYTHONPATH=. python train_qt.py --data-path  data/airlines_processed.csv --pre-epoch 10 --view1-col first_utterance --view2-col context --scenarios view1
```
- to train on askubuntu, replace `airlines` with `askubuntu` in the above command-line, and remove the `--*-col` command-line options
