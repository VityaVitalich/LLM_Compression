#!/bin/bash

MODEL_PATH="/home/cache/llama7b_4w_16a_128fp"
OUTPUT_PATH="logs/llama7b-4w"

lm_eval \
--model hf \
--model_args pretrained=$MODEL_PATH \
--tasks hellaswag,swag,winogrande,boolq,xwinograd_en \
--device cuda:0 \
--batch_size 16 \
--output_path $OUTPUT_PATH
