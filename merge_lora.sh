#!/bin/bash

BASE_MODEL="meta-llama/Llama-2-13b-hf"
ADAPTERS_PATH="/home/LLM_Compression/logs/fine_tuning/lora/Llama13b_lima_4bit_lora/"
SAVE_DIR="/home/LLM_Compression/logs/fine_tuning/full/Llama13b_lima_4bit_lora/"
TOKEN=$HF_TOKEN

python merge_lora.py \
--base_model=$BASE_MODEL \
--adapters_path=$ADAPTERS_PATH \
--save_dir=$SAVE_DIR \
--token=$TOKEN
