import torch
from pathlib import Path
from argparse import ArgumentParser
import os
import numpy as np
from copy import deepcopy
from tqdm import tqdm

if __name__ == '__main__':
    projectors = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

    logs_path = Path("./logs/")
    logs_path.mkdir(parents=True, exist_ok=True)

    for proj_name in tqdm(projectors):
        
        new_model_path = f'/home/data/compression/quik_cache/llama7b_4w_16a_128fp_true_2b-{proj_name}.pt'
        command = f'lm_eval --model hf     --model_args pretrained=meta-llama/Llama-2-7b-hf,pretrained_path={new_model_path} \
        --tasks hellaswag,swag,winogrande,boolq,xwinograd_en    --device cuda:0     --batch_size 16  --output_path logs/llama7b_2b-{proj_name}'
        os.system(command)

