#!/bin/bash

export CUDA_VISIBLE_DEVICES=6

projectname=sota16
dataset=oxford_flowers
cfg=vit_b16_ep50
csc=True
seed=3

{
    bash scripts/alvlm/mains.sh -c ${csc} -w ${projectname} ${dataset} ${cfg} badge AS ${seed}

    bash scripts/alvlm/mains.sh -c ${csc} -w ${projectname} ${dataset} ${cfg} badge AE ${seed}
} 2>&1 | tee logs/${projectname}_${dataset}_${cfg}_seed${seed}.log

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

