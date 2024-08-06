#!/bin/bash

export CUDA_VISIBLE_DEVICES=4

projectname=sota
dataset=fgvc_aircraft
cfg=vit_b32
seed=2

#bash scripts/alvlm/mains.sh -w ${projectname} ${dataset} ${cfg} coreset AS ${seed}
{
    bash scripts/alvlm/mains.sh -w ${projectname} ${dataset} ${cfg} coreset none ${seed}

    bash scripts/alvlm/mains.sh -w ${projectname} ${dataset} ${cfg} badge none ${seed}

    bash scripts/alvlm/mains.sh -w ${projectname} ${dataset} ${cfg} coreset AE ${seed}

    bash scripts/alvlm/mains.sh -w ${projectname} ${dataset} ${cfg} badge AS ${seed}

    bash scripts/alvlm/mains.sh -w ${projectname} ${dataset} ${cfg} badge AE ${seed}
} 2>&1 | tee logs/${projectname}_${dataset}_${cfg}_seed${seed}.log
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

