#!/bin/bash


WANDB_PROJECT_NAME=None
CSC=False

while getopts w:c: flag
do
    case "${flag}" in
        w) WANDB_PROJECT_NAME=${OPTARG};;
        c) CSC=${OPTARG};;
    esac
done

shift $((OPTIND-1))


# custom config
DATASET=$1
CFG=$2  # config file
ALMETHOD=$3 # Active learning method (random, entropy, coreset, badge)
MODE=$4 # [none, AS, AE]


DATA=/hdd/hdd3/jsh/DATA 

TRAINER=ALVLM
CTP="end"  # class token position (end or middle)
NCTX=16  # number of context tokens
SHOTS=-1  # number of shots (1, 2, 4, 8, 16)

for SEED in 1 2 3
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}_al${ALMETHOD}_mode${MODE}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    elif [ "$MODE" = "AS" ]; then 
        python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            TRAINER.COOP.N_CTX ${NCTX} \
            TRAINER.COOP.CSC ${CSC} \
            TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
            DATASET.NUM_SHOTS ${SHOTS} \
            TRAINER.COOPAL.METHOD ${ALMETHOD} \
            TRAINER.COOPAL.ASPATH ${DATASET}.json \
            WANDB_PROJECT_NAME ${WANDB_PROJECT_NAME} \
            TRAINER.COOPAL.GAMMA 0.1
    elif [ "$MODE" = "AE" ]; then 
        python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            TRAINER.COOP.N_CTX ${NCTX} \
            TRAINER.COOP.CSC ${CSC} \
            TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
            DATASET.NUM_SHOTS ${SHOTS} \
            TRAINER.COOPAL.METHOD ${ALMETHOD} \
            TRAINER.COOPAL.AEPATH ${DATASET}.json \
            WANDB_PROJECT_NAME ${WANDB_PROJECT_NAME} \
            TRAINER.COOPAL.GAMMA 0.1
    elif [ "$MODE" = "none" ]; then 
        python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            TRAINER.COOP.N_CTX ${NCTX} \
            TRAINER.COOP.CSC ${CSC} \
            TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
            DATASET.NUM_SHOTS ${SHOTS} \
            TRAINER.COOPAL.METHOD ${ALMETHOD} \
            WANDB_PROJECT_NAME ${WANDB_PROJECT_NAME} \
            TRAINER.COOPAL.GAMMA 0.1
    else 
        echo "MODE should be selected in [none, AS, AE]"
    fi 
done