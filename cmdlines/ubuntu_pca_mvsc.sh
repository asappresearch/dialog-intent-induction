#!/bin/bash

export PYTHONPATH=.

python train_pca.py --model mvsc --data-path data/askubuntu_processed.csv --mvsc-no-unk
