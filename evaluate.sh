#!/bin/bash

MODEL_PATH="/home/LLM_Compression/logs/fine_tuning/full/Llama13b_lima_lora/checkpoint-362"
OUTPUT_PATH="logs/Llama13b_lima_lora_checkpoint-363/"

lm_eval \
--model hf \
--model_args pretrained=$MODEL_PATH,parallelize=True \
--tasks hellaswag,winogrande,boolq,ai2_arc,piqa \
--batch_size 16 \
--output_path $OUTPUT_PATH
