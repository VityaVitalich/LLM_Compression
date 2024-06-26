#!/bin/bash

python /home/Quantization/admm-pruning/main.py \
    --model /home/exp_results/claq/llama2_7b_claq3bit \
    --prune_method admm \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save /home/exp_results/admm_pruning/unstructured/llama2_7b_claq3bit_admm_0@50 \
    --save_model /home/exp_results/admm_pruning/unstructured/llama2_7b_claq3bit_admm_0@50