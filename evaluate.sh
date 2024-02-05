#!/bin/bash

MODEL_PATH="logs/fine_tuning/full/clip_2e-2/checkpoint-7000"
OUTPUT_PATH="logs/llama7b_clip_2e-2_long"

lm_eval \
--model hf \
--model_args pretrained=$MODEL_PATH \
--tasks hellaswag \
--device cuda:0 \
--batch_size 16 \
--output_path $OUTPUT_PATH
