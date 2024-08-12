#!/bin/bash

#SBATCH -J dtd_eurosat       # name of job
#SBATCH -c 4                        # number of cpus required per task
#SBATCH -a 1-9                      # job array index values
#SBATCH -D /home/jin749/jinpcb      # set working directory for batch script
#SBATCH -t 0-03:00:00               # time limit
#SBATCH -o /home/jin749/jinpcb/sbatch/slogs/dtd_eurosat_%A_%a.out    # file for batch script's standard output
#SBATCH -p A5000                    # partition requested

#SBATCH --mem-per-gpu=10G           # memory required per allocated GPU
#SBATCH --gres=gpu:1                # number of gpus required
#SBATCH --partition=A5000           

config=sbatch/config

WANDB_PROJECT_NAME=entropy_filterlr_adam
WANDB_ENTITY=apl_postech
WARM_START=True
FILTER=True
FILTER_LR=$(awk -v task_id=$SLURM_ARRAY_TASK_ID 'NR==task_id {print $3}' $config)
FILTER_OPTIM_NAME=sgd
ALMETHOD_FOR_FILTER=False
CSC=True

DATASET=(awk -v task_id=$SLURM_ARRAY_TASK_ID 'NR==task_id {print $1}' $config)
CFG=vit_b32
ALMETHOD=(awk -v task_id=$SLURM_ARRAY_TASK_ID 'NR==task_id {print $2}' $config)
MODE=AS
SEED=(awk -v task_id=$SLURM_ARRAY_TASK_ID 'NR==task_id {print $4}' $config)


DATA=/home/jin749/DATA

TRAINER=ALVLM
CTP="end"  # class token position (end or middle)
NCTX=16  # number of context tokens
SHOTS=-1  # number of shots (1, 2, 4, 8, 16)


DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}_al${ALMETHOD}_mode${MODE}_warm${WARM_START}_filter${FILTER}_${FILTER_OPTIM_NAME}${FILTER_LR}_filtermethod${ALMETHOD_FOR_FILTER}/seed${SEED}
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
