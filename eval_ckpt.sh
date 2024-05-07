#!/bin/bash

MODEL_PATH="/home/LLM_Compression/ckpts/full/Llama8b_tulu_dora/"
SAVE_DIR="/home/LLM_Compression/logs/Llama8b_tulu_dora/"
TASK="paper"

python eval_checkpoints.py \
--model_dir=$MODEL_PATH \
--save_dir=$SAVE_DIR \
--task=$TASK
