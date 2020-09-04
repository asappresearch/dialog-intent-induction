#!/bin/bash

export PYTHONPATH=.

python train.py --pre-epoch 10 --pre-model qt --num-epochs 0 --data-path data/askubuntu_processed.csv \
    --save-model-path data/ubuntu_qt.pth
