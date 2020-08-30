#!/bin/bash

if [[ ! -f data/ubuntu_qt.pth ]]; then {
    echo Please train askubuntu qt first
} fi

python train.py --pre-epoch 0 --num-epochs 50 --data-path data/askubuntu_processed.csv \
    --model-path data/ubuntu_qt.pth
