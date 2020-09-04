#!/bin/bash

export PYTHONPATH=.

python train_pca.py --data-path data/askubuntu_processed.csv
