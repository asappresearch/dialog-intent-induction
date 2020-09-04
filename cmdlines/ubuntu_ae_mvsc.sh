#!/bin/bash

export PYTHONPATH=.

if [[ ! -f data/ubuntu_ae.pth ]]; then {
    echo Please train askubuntu ae first
    exit 0
} fi

python run_mvsc.py --ref ubuntu_ae_mvsc --model-path data/ubuntu_ae.pth \
    --data-path data/askubuntu_processed.csv --mvsc-no-unk
