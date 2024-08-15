#!/bin/bash

#SBATCH -J nowarm_filterlr_big      # name of job
#SBATCH -c 4                        # number of cpus required per task
#SBATCH --gres=gpu:1                # number of gpus required
#SBATCH -D /home/jin749/jinpcb      # set working directory for batch script
#SBATCH -o /home/jin749/jinpcb/sbatch/slogs/%x_%A_%a.out    # file for batch script's standard output

#SBATCH --mem-per-gpu=40G           # memory required per allocated GPU
#SBATCH -t 0-16:00:00               # time limit
#SBATCH -p A6000                    # partition requested
#SBATCH -a 1-6                      # job array index values
config=sbatch/failed2.csv
conda activate pcb


WARM_START=False
FILTER=True
FILTER_LR=$(awk -F '[,]' -v task_id=$SLURM_ARRAY_TASK_ID 'NR==task_id {print $3}' $config)
FILTER_OPTIM_NAME=sgd
ALMETHOD_FOR_FILTER=False
CSC=True

DATASET=$(awk -F '[,]' -v task_id=$SLURM_ARRAY_TASK_ID 'NR==task_id {print $1}' $config)
CFG=vit_b32
ALMETHOD=$(awk -F '[,]' -v task_id=$SLURM_ARRAY_TASK_ID 'NR==task_id {print $2}' $config)
MODE=AS
SEED=$(awk -F '[,]' -v task_id=$SLURM_ARRAY_TASK_ID 'NR==task_id {print $4}' $config)

WANDB_PROJECT_NAME=${ALMETHOD}_warm${WARM_START}_filter${FILTER}
WANDB_ENTITY=apl_postech


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
