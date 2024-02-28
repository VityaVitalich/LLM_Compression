#!/bin/bash

MODEL_PATH="/home/LLM_Compression/logs/fine_tuning/full/4w_ste_learnable_sft/"
SAVE_DIR="/home/LLM_Compression/logs/4w_ste_sft"

python eval_checkpoints.py \
--model_dir=$MODEL_PATH \
--save_dir=$SAVE_DIR
