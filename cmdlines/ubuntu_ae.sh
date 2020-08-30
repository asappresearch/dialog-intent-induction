#!/bin/bash

python train.py --pre-epoch 20 --pre-model ae --num-epochs 0 \
    --data-path data/askubuntu_processed.csv \
    --save-model-path data/ubuntu_ae.pth
