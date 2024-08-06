#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

projectname=sota
dataset=caltech101
cfg=vit_b32
seed=3

bash scripts/alvlm/mains.sh -w ${projectname} ${dataset} ${cfg} coreset none ${seed}

bash scripts/alvlm/mains.sh -w ${projectname} ${dataset} ${cfg} badge none ${seed}

bash scripts/alvlm/mains.sh -w ${projectname} ${dataset} ${cfg} coreset AS ${seed}

bash scripts/alvlm/mains.sh -w ${projectname} ${dataset} ${cfg} coreset AE ${seed}

bash scripts/alvlm/mains.sh -w ${projectname} ${dataset} ${cfg} badge AS ${seed}

bash scripts/alvlm/mains.sh -w ${projectname} ${dataset} ${cfg} badge AE ${seed}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

