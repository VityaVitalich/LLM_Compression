#!/bin/bash

BASE_MODEL="meta-llama/Llama-2-13b-hf"
ADAPTERS_PATH="/home/LLM_Compression/ckpts/lora/Llama13b_tulu_dora_lowlr/"
SAVE_DIR="/home/LLM_Compression/ckpts/full/Llama13b_tulu_dora_lowlr/"
TOKEN=$HF_TOKEN

python merge_lora.py \
--base_model=$BASE_MODEL \
--adapters_path=$ADAPTERS_PATH \
--save_dir=$SAVE_DIR \
--token=$TOKEN
