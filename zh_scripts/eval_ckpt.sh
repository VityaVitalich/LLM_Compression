#!/bin/bash

#SBATCH --job-name=llmcompr

#SBATCH --partition=ais-gpu

#SBATCH --mail-type=ALL

#SBATCH --mail-user=V.Moskvoretskii@skoltech.ru

#SBATCH --output=zh_logs/eval_ckpt.txt
#SBATCH --time=0-10

#SBATCH --mem=16G

#SBATCH --nodes=1

#SBATCH -c 8

#SBATCH --gpus=4

srun singularity exec --bind /trinity/home/v.moskvoretskii/:/home -f --nv /trinity/home/v.moskvoretskii/images/glm.sif bash -c '
    ls;
    cd /home;
    ls;
    export HF_TOKEN=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;
    export SAVING_DIR=/home/cache/;
    export WANDB_API_KEY=2b740bffb4c588c7274a6e8cf4e39bd56344d492;
    export CUDA_LAUNCH_BLOCKING=1;
    export HF_HOME=/home/cache/;
    cd /home/LLM_Compression;
    ls;
    nvidia-smi;
    pip list;
    sh eval_ckpt.sh;
'

