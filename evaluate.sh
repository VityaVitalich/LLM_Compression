#!/bin/bash

MODEL_PATH="/home/LLM_Compression/ckpts/full/Llama13b_tulu_dora_lowlr/checkpoint-500/"
OUTPUT_PATH="logs/Llama13b_tulu_dora_lowlr_ckpt500/"

lm_eval \
--model hf \
--model_args pretrained=$MODEL_PATH,parallelize=True \
--tasks hellaswag,winogrande,boolq,ai2_arc,piqa \
--batch_size 16 \
--output_path $OUTPUT_PATH
