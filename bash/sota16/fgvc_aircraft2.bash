#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

projectname=sota16
dataset=fgvc_aircraft
cfg=vit_b16_ep50
seed=2

{
    bash scripts/alvlm/mains.sh -w ${projectname} ${dataset} ${cfg} coreset AS ${seed}

    bash scripts/alvlm/mains.sh -w ${projectname} ${dataset} ${cfg} coreset AE ${seed}

    bash scripts/alvlm/mains.sh -w ${projectname} ${dataset} ${cfg} badge AS ${seed}

    bash scripts/alvlm/mains.sh -w ${projectname} ${dataset} ${cfg} badge AE ${seed}
} 2>&1 | tee logs/${projectname}_${dataset}_${cfg}_seed${seed}.log

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

