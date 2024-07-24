#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

projectname=sota
dataset=dtd
cfg=vit_b32

bash scripts/alvlm/main2.sh -w ${projectname} ${dataset} ${cfg} coreset none

bash scripts/alvlm/main2.sh -w ${projectname} ${dataset} ${cfg} coreset AS

bash scripts/alvlm/main2.sh -w ${projectname} ${dataset} ${cfg} coreset AE

bash scripts/alvlm/main2.sh -w ${projectname} ${dataset} ${cfg} badge none

bash scripts/alvlm/main2.sh -w ${projectname} ${dataset} ${cfg} badge AS

bash scripts/alvlm/main2.sh -w ${projectname} ${dataset} ${cfg} badge AE

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

