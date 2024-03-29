#!/bin/bash

MODEL_PATH="/home/LLM_Compression/logs/fine_tuning/full/Llama7b_ste_lima_4bit/"
SAVE_DIR="/home/LLM_Compression/logs/Llama7b_ste_lima_4bit/"
TASK="paper"

python eval_checkpoints.py \
--model_dir=$MODEL_PATH \
--save_dir=$SAVE_DIR \
--task=$TASK
