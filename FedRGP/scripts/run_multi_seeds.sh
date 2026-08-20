#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 5 ]; then
    echo "Usage: $0 <GPU_ID> [SEEDS] [DATASETS] [CONFIG] [SHOTS]"
    echo "SEEDS: comma-separated values, e.g. 0,1,2"
    echo "DATASETS: comma-separated names from caltech101,dtd,fgvc_aircraft,food101,oxford_flowers,oxford_pets,ucf101"
    exit 1
fi

GPU=$1
SEEDS_INPUT=${2:-0}
DATASETS_INPUT=${3:-caltech101,dtd,fgvc_aircraft,food101,oxford_flowers,oxford_pets,ucf101}
CONFIG_FILE=${4:-base2novel_vit_b16}
SHOTS=${5:-16}
VALID_DATASETS=" caltech101 dtd fgvc_aircraft food101 oxford_flowers oxford_pets ucf101 "

IFS=',' read -ra SEEDS <<< "$SEEDS_INPUT"
IFS=',' read -ra DATASETS <<< "$DATASETS_INPUT"

for DATASET in "${DATASETS[@]}"; do
    if [[ "$VALID_DATASETS" != *" $DATASET "* ]]; then
        echo "Unsupported dataset: $DATASET"
        exit 1
    fi
done

for SEED in "${SEEDS[@]}"; do
    export SEED
    for DATASET in "${DATASETS[@]}"; do
        bash scripts/FedRGP/base2novel_train.sh "$GPU" "$DATASET" "$SHOTS" "$CONFIG_FILE"
        bash scripts/FedRGP/base2novel_test.sh "$GPU" "$DATASET" base "$SHOTS" "$CONFIG_FILE"
        bash scripts/FedRGP/base2novel_test.sh "$GPU" "$DATASET" new "$SHOTS" "$CONFIG_FILE"
    done
done
