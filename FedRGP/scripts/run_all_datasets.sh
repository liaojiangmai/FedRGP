#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <GPU_ID> [CONFIG] [SHOTS]"
    exit 1
fi

GPU=$1
CONFIG_FILE=${2:-base2novel_vit_b16}
SHOTS=${3:-16}
DATASETS=(caltech101 dtd fgvc_aircraft food101 oxford_flowers oxford_pets ucf101)

for DATASET in "${DATASETS[@]}"; do
    bash scripts/FedRGP/base2novel_train.sh "$GPU" "$DATASET" "$SHOTS" "$CONFIG_FILE"
    bash scripts/FedRGP/base2novel_test.sh "$GPU" "$DATASET" base "$SHOTS" "$CONFIG_FILE"
    bash scripts/FedRGP/base2novel_test.sh "$GPU" "$DATASET" new "$SHOTS" "$CONFIG_FILE"
done
