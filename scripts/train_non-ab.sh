#!/bin/bash

for i in $(seq 1 12);
do
    CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --nproc_per_node=4 --master_port=29519 train_non-ab.py --learning_rate 1e-4 --weight_decay 0.02 --batch_size 24 --gpus 4 --max_epochs 100 --downsample_rate $i
    sleep 1m
done
