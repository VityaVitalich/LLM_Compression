#!/bin/bash

MODEL_PATH="allenai/tulu-2-dpo-70b"
OUTPUT_PATH="/home/cache/lima_tulu70b-dpo/" #"/home/cache/wikitext_llama-7b"
BATCH_SIZE=4
DATASET_NAME="VityaVitalich/LIMA"
DATASET_CONFIG_NAME="wikitext-2-raw-v1"
BLOCK_SIZE=2048
DATASET_PERCENTAGE=100
CACHE_DIR="/home/cache/"
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

