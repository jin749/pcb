#!/bin/bash

export CUDA_VISIBLE_DEVICES=4

projectname=sota16
dataset=caltech101
cfg=vit_b16_ep50

bash scripts/alvlm/main.sh ${dataset} ${cfg} coreset none

# bash scripts/alvlm/main.sh -w ${projectname} ${dataset} ${cfg} badge none

# bash scripts/alvlm/main.sh -w ${projectname} ${dataset} ${cfg} coreset AS

# bash scripts/alvlm/main.sh -w ${projectname} ${dataset} ${cfg} coreset AE

# bash scripts/alvlm/main.sh -w ${projectname} ${dataset} ${cfg} badge AS

# bash scripts/alvlm/main.sh -w ${projectname} ${dataset} ${cfg} badge AE

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

