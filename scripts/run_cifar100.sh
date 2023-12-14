#!/bin/bash
# bash scripts/run_cifar100.sh 
set -e
set -x

#export CUDA_VISIBLE_DEVICES=2,3,4,5,6 

torchrun train_mp.py \
    --dataset_name 'cifar100' \
    --batch_size 256 \
    --grad_from_block 11 \
    --epochs 200 \
   --num_workers 8 \
    --use_ssb_splits \
    --sup_weight 0.35 \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr 0.1 \
    --eval_funcs 'v2' \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --memax_weight 4 \
    --exp_name cifar100_simgcd
