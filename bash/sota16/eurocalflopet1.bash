#!/bin/bash

export CUDA_VISIBLE_DEVICES=4

projectname=sota16
cfg=vit_b16_ep50
seed=1
dataset=eurocalflopet

{

    bash scripts/alvlm/mains.sh -w ${projectname} eurosat ${cfg} coreset AE ${seed}

    bash scripts/alvlm/mains.sh -w ${projectname} caltech101 ${cfg} badge AS ${seed}

    bash scripts/alvlm/mains.sh -w ${projectname} oxford_flowers ${cfg} badge AS ${seed}

    bash scripts/alvlm/mains.sh -w ${projectname} oxford_pets ${cfg} badge AE ${seed}
    
} 2>&1 | tee logs/${projectname}_${dataset}_${cfg}_seed${seed}.log

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

