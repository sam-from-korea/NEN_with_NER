#!/bin/bash

python3 run_ner.py \
    --data_dir ./../dataset \
    --labels labels.txt \
    --model_name_or_path ./../pretrained_model \
    --output_dir output/nano \
    --max_seq_length 192 \
    --num_train_epochs 30 \
    --per_device_train_batch_size 32 \
    --save_steps 2000 \
    --seed 1 \
    --do_train \
    --do_eval \
    --overwrite_output_dir