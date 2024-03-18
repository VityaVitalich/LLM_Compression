#!/bin/bash

MODEL_PATH="/home/LLM_Compression/logs/fine_tuning/full/Llama13b_lima/"
SAVE_DIR="/home/LLM_Compression/logs/Llama13b_lima/"
TASK="paper"

python eval_checkpoints.py \
--model_dir=$MODEL_PATH \
--save_dir=$SAVE_DIR \
--task=$TASK
