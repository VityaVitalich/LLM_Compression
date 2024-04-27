#!/bin/bash

BASE_MODEL="/home/cache/Llama7b_3bit_128fp_owq_max_3/"
ADAPTERS_PATH="/home/LLM_Compression/ckpts/lora/Llama7b_tulu_quik_3bit_lora/"
SAVE_DIR="/home/LLM_Compression/ckpts/full/Llama7b_tulu_quik_3bit_lora/"
TOKEN=$HF_TOKEN

python merge_lora.py \
--base_model=$BASE_MODEL \
--adapters_path=$ADAPTERS_PATH \
--save_dir=$SAVE_DIR \
--token=$TOKEN
