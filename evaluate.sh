#!/bin/bash

MODEL_PATH="logs/fine_tuning/full/clip_2e-2_zero/checkpoint-6000"
OUTPUT_PATH="logs/llama7b_clip_2e-2_zero"

lm_eval \
--model hf \
--model_args pretrained=$MODEL_PATH \
--tasks hellaswag \
--device cuda:0 \
--batch_size 16 \
--output_path $OUTPUT_PATH
