#!/bin/bash

BASE_MODEL="ZhMax/Llama-3-8B-test"
ADAPTERS_PATH="/home/LLM_Compression/ckpts/lora/Llama8b_tulu_dora/"
SAVE_DIR="/home/LLM_Compression/ckpts/full/Llama8b_tulu_dora/"
TOKEN=$HF_TOKEN

python merge_lora.py \
--base_model=$BASE_MODEL \
--adapters_path=$ADAPTERS_PATH \
--save_dir=$SAVE_DIR \
--token=$TOKEN
