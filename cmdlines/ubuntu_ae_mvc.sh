#!/bin/bash

export PYTHONPATH=.

if [[ ! -f data/ubuntu_ae.pth ]]; then {
    echo Please train askubuntu ae first
} fi

python train.py --pre-epoch 0 --num-epochs 50 --data-path data/askubuntu_processed.csv \
    --model-path data/ubuntu_ae.pth
