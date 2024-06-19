import torch
from pathlib import Path
from argparse import ArgumentParser
import os
import numpy as np
from copy import deepcopy
from tqdm import tqdm

if __name__ == '__main__':
    lora_ckpts = ['174', '349', '523', '698', '870']

    logs_path = Path("./logs/")
    logs_path.mkdir(parents=True, exist_ok=True)

    for ckpt_name in tqdm(lora_ckpts):
        
        new_model_path = f'/home/data/compression/clip_sm_cache/fine_tuning/lora/ckpt{ckpt_name}_sm_gamma-2e-2'
        command = f'lm_eval --model hf     --model_args pretrained={new_model_path} \
        --tasks hellaswag,swag,winogrande,boolq,xwinograd_en    --device cuda:0     --batch_size 4  --output_path logs/llama7b_clip-lora-{ckpt_name}'
        os.system(command)

    full_ckpt = [43, 87, 130, 174, 215]
    for ckpt_name in tqdm(full_ckpt):
        
        new_model_path = f'/home/data/compression/clip_sm_cache/fine_tuning/full/checkpoint-{ckpt_name}'
        command = f'lm_eval --model hf     --model_args pretrained={new_model_path} \
        --tasks hellaswag,swag,winogrande,boolq,xwinograd_en    --device cuda:0     --batch_size 4  --output_path logs/llama7b_clip-full-{ckpt_name}'
        os.system(command)