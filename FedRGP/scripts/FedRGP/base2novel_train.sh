#!/bin/bash

# Set COOP_DATASET before running, e.g. export COOP_DATASET=/path/to/datasets
if [ -z "${COOP_DATASET}" ] || [ ! -d "${COOP_DATASET}" ] || [ "${COOP_DATASET}" = "/path/to/datasets" ]; then
  export COOP_DATASET=/path/to/datasets
fi

GPU=$1                                    # GPU ID
DATASET=$2                                # Dataset name
SHOTS=${3:-8}                             # Number of shots, default 8
CONFIG_FILE=${4:-"base2novel_vit_b16"}    # Config file name
SEED=${SEED:-${5:-0}}
NOISE_RATIO=${NOISE_RATIO:-${6:-}}
NOISE_TYPE=${NOISE_TYPE:-${7:-}}

DATA="${COOP_DATASET}"                    # Dataset root path
cfg_file=${CONFIG_FILE}
trainer=FedRGP
model=FedRGP
SUBSAMPLE_CLASSES=base
USEALL=False

DIR=output/${DATASET}/${trainer}/${cfg_file}/${SHOTS}shots/seed${SEED}/${SUBSAMPLE_CLASSES}

if [ -d "$DIR" ]; then
  echo "Warning: Directory ${DIR} exists, will overwrite"
fi

mkdir -p ${DIR}

EXTRA_OPTS=()
if [ -n "${NOISE_RATIO}" ]; then
  EXTRA_OPTS+=(DATASET.LABEL_NOISE_RATIO ${NOISE_RATIO})
fi
if [ -n "${NOISE_TYPE}" ]; then
  EXTRA_OPTS+=(DATASET.LABEL_NOISE_TYPE ${NOISE_TYPE})
fi

CUDA_VISIBLE_DEVICES=${GPU} python federated_main.py \
--root ${DATA} \
--output-dir ${DIR} \
--seed ${SEED} \
--model ${model} \
--trainer ${trainer} \
--config-file configs/trainers/${trainer}/${cfg_file}.yaml \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--num_shots ${SHOTS} \
--debug-mode true \
DATASET.SUBSAMPLE_CLASSES ${SUBSAMPLE_CLASSES} \
DATASET.USEALL ${USEALL} \
${EXTRA_OPTS[@]}
