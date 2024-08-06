#!/bin/bash

export CUDA_VISIBLE_DEVICES=6

projectname=sota16
dataset=stanford_cars
cfg=vit_b16_ep50
seed=1

{

    bash scripts/alvlm/mains.sh -w ${projectname} ${dataset} ${cfg} badge AE ${seed}
    
} 2>&1 | tee logs/${projectname}_${dataset}_${cfg}_seed${seed}.log

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7