#!/bin/bash

MODEL_PATH="logs/fine_tuning/full/ste_learnable/checkpoint-4000/"
OUTPUT_PATH="logs/llama7b-4b-128fp-ste_learnable"

lm_eval \
--model hf \
--model_args pretrained=$MODEL_PATH \
--tasks hellaswag,swag,winogrande,boolq,xwinograd_en \
--device cuda:0 \
--batch_size 16 \
--output_path $OUTPUT_PATH
