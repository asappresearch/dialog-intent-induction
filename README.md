Code and data for paper ["Dialog Intent Induction with Deep Multi-View Clustering"](https://arxiv.org/abs/1908.11487), Hugh Perkins and Yi Yang, 2019, to appear in EMNLP 2019.

Data is available in the sub-directory [data](data), with a specific [LICENSE](data/LICENSE) file.

# Dialog Intent Induction

## Designing user intents is hard

- Existing dialog classification work assumes that target intents are given
- In industry, a classifier may have hundreds of intents
- their discovery involves considerable human effort

## Automatic discovery of dialog intents from data

- We aim to automatically cluster conversations into intents
- A key challenge is to learn an effective conversation encoder, in an unsupervised way
- ... such that distance in embedding space relates closely to human perceived semantic distance

<img src="images/intents_acad_vs_indust.png" width="500" />

# Multiview Clustering

Consider two dialogs for FindAirPods:

<img src="images/example_dialogs.png" width="350" />

- The user query utterances are lexically and syntactically dissimilar
- The solution trajectories are similar
- We leverage the solution trajectory as a weak supervision signal

<img src="images/multiview_clustering.png" width="350" />

- Multi-view clustering seeks to reconcile the clusters from multiple views
- However classic multi-view clustering uses fixed encoders

# Deep Multiview Clustering using AV-KMeans

We propose the AV-Kmeans algorithm, to jointly learn the encoder and the cluster assignment:

<img src="images/avkmeans_pseudocode.png" width="350" /> &nbsp; <img src="images/avkmeans_graph.png" width="350" />

- The encoder is fixed for clustering, and updated for classification
- we use prototypical loss for the classification task

# New datasets

- Existing goal-oriented dialog datasets concern predefined dialog intents
- We propose two new datasets, drawn from real-world distributions, for evaluating intent induction

## Twitter Airlines Customer Support (TwACS)

- We use dialogs in the airline industry from the Twitter Support dataset released by Kaggle
- The training set comprises 43,072 unlabeled dialogs. We annotated 500 dialogs to create a test set

## AskUbuntu duplicate detection

- We reprocessed the AskUbuntu question duplicate dataset into an intent clustering dataset
- The original data comprises a graph of questions, connected by human duplicate annotations
- Our training dataset comprises the unlabeled 4,692 questions and answers from the 20 largest connected components

# Experiments

We experiment with three types of unsupervised pre-training:

- none/PCA
- auto-encoder
- quick thoughts

We compare with two popular clustering algorithms:

- k-means
- [Multi-View Spectral Clustering (MVSC)](https://github.com/mariceli3/multiview)

<img src="images/results_twacs.png" width="350" /> <img src="images/results_askubuntu.png" width="350" />

- AV-Kmeans outperforms the baselines for each pretraining algorithm
- AV-Kmeans with quick thoughts pre-training shows the strongest performance

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

