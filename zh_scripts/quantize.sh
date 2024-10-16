#!/bin/bash

#SBATCH --job-name=llmcompr

#SBATCH --partition=ais-gpu

#SBATCH --mail-type=ALL

#SBATCH --mail-user=V.Moskvoretskii@skoltech.ru

#SBATCH --output=zh_logs/quantize.txt
#SBATCH --time=0-05

#SBATCH --mem=32G

#SBATCH --nodes=1

#SBATCH -c 8

#SBATCH --gpus=1

srun singularity exec --bind /trinity/home/v.moskvoretskii/:/home -f --nv /trinity/home/v.moskvoretskii/images/compression.sif bash -c '
    cd /home;
    export HF_TOKEN=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;
    export SAVING_DIR=/home/cache/;
    export HF_HOME=/home/cache/;
    export TRANSFORMERS_CACHE=/home/cache/;
    export WANDB_API_KEY=2b740bffb4c588c7274a6e8cf4e39bd56344d492;
    export CUDA_LAUNCH_BLOCKING=1;
    cd /home/LLM_Compression/QUIK/experiments/fake_quant;
    nvidia-smi;
    pip list;
    python llama.py --model ZhMax/Llama-3-8B-test --path_to_act_scales /home/LLM_Compression/QUIK/experiments/act_scales/llama3_8b_obs_w2_ptb_max.pt --path_to_save_quant_model /home/cache/Llama8b_2bit_128fp_owq_max_2 --fp_features 128 --a_bits 16 --w_bits 2 --w_clip --dataset wikitext2 --hf_token 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;
'

