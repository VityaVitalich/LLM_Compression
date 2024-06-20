import torch
from pathlib import Path
from argparse import ArgumentParser
import os
import numpy as np
from copy import deepcopy
from tqdm import tqdm

if __name__ == '__main__':
    all_models = os.listdir('/home/cache/llama7b_svd/')
    save_logs_path = Path("/home/LLM_Compression/logs/svd_eval/")
    save_logs_path.mkdir(parents=True, exist_ok=True)

    for model_path in tqdm(all_models):
        
        new_model_path = f'/home/cache/llama7b_svd/{model_path}'
        save_path = f'/home/LLM_Compression/logs/{model_path}'
        command = f'lm_eval --model hf     --model_args pretrained={new_model_path} \
        --tasks hellaswag    --device cuda:0     --batch_size 16  --output_path {save_path}'
        os.system(command)

