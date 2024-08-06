#!/bin/bash

export CUDA_VISIBLE_DEVICES=4

projectname=sota16
cfg=vit_b16_ep50
seed=3
dataset=eurocalflopet

{

    bash scripts/alvlm/mains.sh -w ${projectname} eurosat ${cfg} coreset AS ${seed}

    bash scripts/alvlm/mains.sh -w ${projectname} caltech101 ${cfg} badge none ${seed}

    bash scripts/alvlm/mains.sh -w ${projectname} oxford_pets ${cfg} badge AE ${seed}

    bash scripts/alvlm/mains.sh -w ${projectname} oxford_pets ${cfg} badge AS ${seed}

} 2>&1 | tee logs/${projectname}_${dataset}_${cfg}_seed${seed}.log

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

