#!/bin/bash
source /home/polaris/chengr/miniconda3/bin/activate venv

for i in {1..5}
do
   python3 run_ner.py \
       --data_dir ./../database/ablation/$i \
       --labels labels.txt \
       --model_name_or_path ./../pretrained_model \
       --output_dir ./../database/ablation/output/$i \
       --max_seq_length 192 \
       --num_train_epochs 3 \
       --per_device_train_batch_size 32 \
       --save_steps 2000 \
       --seed 1 \
       --do_train \
       --do_eval \
       --overwrite_output_dir
done
