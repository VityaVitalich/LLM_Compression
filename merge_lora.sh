#!/bin/bash

BASE_MODEL="/home/LLM_Compression/logs/fine_tuning/full/Llama13b_lima/"
ADAPTERS_PATH="/home/LLM_Compression/logs/fine_tuning/full/Llama13b_lima/"
SAVE_DIR="/home/LLM_Compression/logs/Llama13b_lima/"
TOKEN=""

python merge_lora.py \
--base_model=$BASE_MODEL \
--adapters_path=$ADAPTERS_PATH \
--save_dir=$SAVE_DIR \
--token=$TOKEN