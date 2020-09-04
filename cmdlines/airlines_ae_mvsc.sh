#!/bin/bash

export PYTHONPATH=.

if [[ ! -f data/airlines_ae.pth ]]; then {
    echo Please train airlines ae first
    exit 0
} fi

python run_mvsc.py --ref airlines_ae_mvsc --model-path data/airlines_ae.pth \
    --data-path data/airlines_processed.csv --num-cluster-samples 100
