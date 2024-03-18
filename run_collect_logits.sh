#!/bin/bash

MODEL_PATH="meta-llama/Llama-2-7b-hf"
OUTPUT_PATH="/home/data/compression/quik_cache/tulu-llama-7b" #"/home/cache/wikitext_llama-7b"
BATCH_SIZE=2
DATASET_NAME="allenai/tulu-v2-sft-mixture"
DATASET_CONFIG_NAME="wikitext-2-raw-v1"
BLOCK_SIZE=2048
DATASET_PERCENTAGE=10
CACHE_DIR="/home/data/taxonomy/hf_cache/"
INSTRUCT=1
TOKEN="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"


python collect_logits.py \
--model_path $MODEL_PATH \
--dataset-name $DATASET_NAME \
--save_dir $OUTPUT_PATH \
--batch_size $BATCH_SIZE \
--dataset_config_name $DATASET_CONFIG_NAME \
--block_size $BLOCK_SIZE \
--dataset_percentage $DATASET_PERCENTAGE \
--cache_dir $CACHE_DIR \
--instruct $INSTRUCT \
--token $TOKEN

