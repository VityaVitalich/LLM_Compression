#!/bin/bash

MODEL_PATH="meta-llama/Llama-2-7b-hf"
OUTPUT_PATH="/home/cache/wikitext_llama-7b"
BATCH_SIZE=16
DATASET_NAME="wikitext"
DATASET_CONFIG_NAME="wikitext-2-raw-v1"
BLOCK_SIZE=1024
DATASET_PERCENTAGE=40
CACHE_DIR="/home/cache/"


python collect_logits.py \
--model_path $MODEL_PATH \
--dataset-name $DATASET_NAME \
--save_dir $OUTPUT_PATH \
--batch_size $BATCH_SIZE \
--dataset_config_name $DATASET_CONFIG_NAME \
--block_size $BLOCK_SIZE \
--dataset_percentage $DATASET_PERCENTAGE \
--cache_dir $CACHE_DIR \

