#!/bin/bash

save_dir="eval_results/"
global_record_file="eval_results/eval_record_collection.csv"
model="/workspace/HuggingFace-Download-Accelerator/llama/models--meta-llama--Llama-3.1-8B-Instruct"
selected_subjects="physics"
gpu_util=0.8

cd ../../
export CUDA_VISIBLE_DEVICES=0

python evaluate_from_local.py \
                 --selected_subjects $selected_subjects \
                 --save_dir $save_dir \
                 --model $model \
                 --global_record_file $global_record_file \
                 --gpu_util $gpu_util

