#!/bin/bash

python train.py --pre-epoch 0 --num-epochs 0 --data-path data/airlines_processed.csv \
    --view1-col first_utterance --view2-col context --label-col tag
