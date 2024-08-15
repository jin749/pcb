#!/bin/bash


WANDB_PROJECT_NAME=None
WANDB_ENTITY=None
WARM_START=False
FILTER=False # only for mode AS
FILTER_LR=None # only for mode AS
FILTER_OPTIM_NAME=sgd # only for mode AS
ALMETHOD_FOR_FILTER=False # only for mode AS
CSC=True

while getopts n:e:c:w:f:l:m:o: flag
do
    case "${flag}" in
        n) WANDB_PROJECT_NAME=${OPTARG};;
        e) WANDB_ENTITY=${OPTARG};;
        c) CSC=${OPTARG};;
        w) WARM_START=${OPTARG};;
        f) FILTER=${OPTARG};;
        l) FILTER_LR=${OPTARG};;
        m) ALMETHOD_FOR_FILTER=${OPTARG};;
        o) FILTER_OPTIM_NAME=${OPTARG};;
    esac
done

shift $((OPTIND-1))


# custom config
DATASET=$1
CFG=$2  # config file
ALMETHOD=$3 # Active learning method (random, entropy, coreset, badge)
MODE=$4 # [none, AS, AE]
SEED=$5


DATA=/hdd/hdd3/jsh/DATA 

TRAINER=ALVLM
CTP="end"  # class token position (end or middle)
NCTX=16  # number of context tokens
SHOTS=-1  # number of shots (1, 2, 4, 8, 16)


DIR=output/${DATASET}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}_al${ALMETHOD}_mode${MODE}_warm${WARM_START}_filter${FILTER}_${FILTER_OPTIM_NAME}${FILTER_LR}_f-method${ALMETHOD_FOR_FILTER}/seed${SEED}
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
        TRAINER.COOPAL.WARM_START ${WARM_START} \
        TRAINER.COOPAL.FILTER ${FILTER} \
        TRAINER.COOPAL.FILTER_LR ${FILTER_LR} \
        TRAINER.COOPAL.FILTER_OPTIM_NAME ${FILTER_OPTIM_NAME} \
        TRAINER.COOPAL.ALMETHOD_FOR_FILTER ${ALMETHOD_FOR_FILTER} \
        WANDB_PROJECT_NAME ${WANDB_PROJECT_NAME} \
        WANDB_ENTITY ${WANDB_ENTITY} \
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
        TRAINER.COOPAL.WARM_START ${WARM_START} \
        WANDB_PROJECT_NAME ${WANDB_PROJECT_NAME} \
        WANDB_ENTITY ${WANDB_ENTITY} \
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
        TRAINER.COOPAL.WARM_START ${WARM_START} \
        WANDB_PROJECT_NAME ${WANDB_PROJECT_NAME} \
        WANDB_ENTITY ${WANDB_ENTITY} \
        TRAINER.COOPAL.GAMMA 0.1
else 
    echo "MODE should be selected in [none, AS, AE]"
fi 
