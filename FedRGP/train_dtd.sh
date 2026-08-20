#!/bin/bash

GPU=${1:-0}
SHOTS=${2:-16}
CONFIG_FILE=${3:-"base2novel_vit_b16"}
SEED=${SEED:-0}
NOISE_RATIO=${NOISE_RATIO:-0.2}
NOISE_TYPE=${NOISE_TYPE:-symmetric}

export SEED NOISE_RATIO NOISE_TYPE

bash scripts/FedRGP/base2novel_train.sh ${GPU} dtd ${SHOTS} ${CONFIG_FILE}

# auto-test
bash scripts/FedRGP/base2novel_test.sh ${GPU} dtd base ${SHOTS} ${CONFIG_FILE}
bash scripts/FedRGP/base2novel_test.sh ${GPU} dtd new ${SHOTS} ${CONFIG_FILE}
