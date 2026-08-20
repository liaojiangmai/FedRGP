#!/bin/bash
set -euo pipefail

if [ $# -lt 2 ] || [ $# -gt 4 ]; then
    echo "Usage: $0 <GPU_ID> <DATASETS> [CONFIG] [SHOTS]"
    echo "DATASETS: space-separated names from caltech101 dtd fgvc_aircraft food101 oxford_flowers oxford_pets ucf101"
    exit 1
fi

GPU=$1
DATASETS_INPUT=$2
CONFIG_FILE=${3:-base2novel_vit_b16}
SHOTS=${4:-16}
VALID_DATASETS=" caltech101 dtd fgvc_aircraft food101 oxford_flowers oxford_pets ucf101 "

for DATASET in $DATASETS_INPUT; do
    if [[ "$VALID_DATASETS" != *" $DATASET "* ]]; then
        echo "Unsupported dataset: $DATASET"
        exit 1
    fi
    bash scripts/FedRGP/base2novel_train.sh "$GPU" "$DATASET" "$SHOTS" "$CONFIG_FILE"
    bash scripts/FedRGP/base2novel_test.sh "$GPU" "$DATASET" base "$SHOTS" "$CONFIG_FILE"
    bash scripts/FedRGP/base2novel_test.sh "$GPU" "$DATASET" new "$SHOTS" "$CONFIG_FILE"
done
