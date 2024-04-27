#!/bin/bash

MODEL_PATH="/home/LLM_Compression/ckpts/full/Llama7b_tulu_quik_2bit_lora_lowlr/"
SAVE_DIR="/home/LLM_Compression/logs/Llama7b_tulu_quik_2bit_lora_lowlr/"
TASK="paper"

python eval_checkpoints.py \
--model_dir=$MODEL_PATH \
--save_dir=$SAVE_DIR \
--task=$TASK
