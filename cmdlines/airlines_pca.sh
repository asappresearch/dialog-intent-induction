#!/bin/bash

export PYTHONPATH=.

python train_pca.py --data-path data/airlines_processed.csv \
    --view1-col first_utterance --view2-col context --label-col tag
