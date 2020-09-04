#!/bin/bash

export PYTHONPATH=.

python train.py --pre-epoch 20 --pre-model ae --num-epochs 0 \
    --data-path data/airlines_processed.csv \
    --view1-col first_utterance --view2-col context --label-col tag \
    --save-model-path data/airlines_ae.pth
