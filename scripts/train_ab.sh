#!/bin/bash

for i in $(seq 1 12);
do
    CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 --master_port=29519 train_ab.py --learning_rate 1e-4 --weight_decay 0.1 --batch_size 24 --gpus 2 --max_epochs 100 --downsample_rate $i
    sleep 1m
done


#CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 train_ab.py --learning_rate 1e-4 --weight_decay 0.0 --batch_size 12 --gpus 2 --max_epochs 100 --downsample_rate 1