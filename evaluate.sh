#!/bin/bash

MODEL_PATH="meta-llama/Llama-2-70b-hf"
OUTPUT_PATH="logs/Llama-70b"

lm_eval \
--model hf \
--model_args pretrained=$MODEL_PATH,parallelize=True \
--tasks hellaswag,winogrande,boolq,ai2_arc,piqa \
--batch_size 4 \
--output_path $OUTPUT_PATH
