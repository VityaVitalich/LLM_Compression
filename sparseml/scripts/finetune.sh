#!/bin/bash

export WANDB_API_KEY=5e5da88bc4c9243bdeeea67927f41590cd612424

accelerate launch \
    --config_file /home/LLM_compression/sparseml/llama2_7b_tutorial/example_fsdp_config.yaml \
    --no_python sparseml.transformers.text_generation.finetune \
    --model /home/LLaMA/huggingface/Llama-2-7b-hf \
    --dataset "wikitext" \
    --dataset_config_name "wikitext-2-raw-v1" \
    --output_dir /home/exp_results/sparseml/tutorial/distill \
    --splits "train" \
    --num_train_epochs 2 \
    --precision "bfloat16" \
    --gradient_checkpointing False \
    --bf16 True \
    --learning_rate 0.00005 \
    --lr_scheduler_type "linear" \
    --max_seq_length 4 \
    --per_device_train_batch_size 1 \
    --max_grad_norm 2 \
    --warmup_steps 20