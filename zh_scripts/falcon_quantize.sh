#!/bin/bash

#SBATCH --job-name=llmcompr

#SBATCH --partition=ais-gpu

#SBATCH --mail-type=ALL

#SBATCH --mail-user=V.Moskvoretskii@skoltech.ru

#SBATCH --output=zh_logs/falcon_quantize.txt
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
    python falcon.py --model tiiuae/falcon-7b --path_to_act_scales /home/LLM_Compression/QUIK/experiments/act_scales/falcon-7b.pt --path_to_save_quant_model /home/cache/falcon7b_3bit --fp_features 128 --a_bits 16 --w_bits 3 --w_clip --dataset wikitext2 --hf_token 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;
'

