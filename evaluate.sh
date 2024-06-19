#!/bin/bash

MODEL_PATH="/home/LLM_Compression/ckpts/full/Llama8b_tulu_ste_2bit/checkpoint-500"
OUTPUT_PATH="logs/Llama8b_tulu_ste_2bit_ckpt500/"

lm_eval \
--model hf \
--model_args pretrained=$MODEL_PATH,parallelize=True \
--tasks hellaswag,winogrande,boolq,ai2_arc,piqa \
--batch_size 16 \
--output_path $OUTPUT_PATH
