#!/bin/bash

MODEL_PATH="gpt2" #"meta-llama/Llama-2-7b-hf"
OUTPUT_PATH="logs/gpt2_wikitext"
BATCH_SIZE=32
DATASET_NAME="wikitext"
DATASET_CONFIG_NAME="wikitext-2-raw-v1"
BLOCK_SIZE=32
DATASET_PERCENTAGE=1
CACHE_DIR="/home/data/taxonomy/hf_cache/"


python collect_logits.py \
--model_path $MODEL_PATH \
--dataset-name $DATASET_NAME \
--save_dir $OUTPUT_PATH \
--batch_size $BATCH_SIZE \
--dataset_config_name $DATASET_CONFIG_NAME \
--block_size $BLOCK_SIZE \
--dataset_percentage $DATASET_PERCENTAGE \
--cache_dir $CACHE_DIR \
