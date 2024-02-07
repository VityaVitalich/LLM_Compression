import torch
from pathlib import Path
from argparse import ArgumentParser
import os
import numpy as np
from copy import deepcopy
from tqdm import tqdm

if __name__ == '__main__':
    layer_range = np.arange(0, 32)

    logs_path = Path("./logs/")
    logs_path.mkdir(parents=True, exist_ok=True)

    for layer_num in tqdm(layer_range):
        
        new_model_path = f'/home/data/compression/quik_cache/llama7b_4w_16a_128fp_true_2b-layer{layer_num}.pt'
        command = f'lm_eval --model hf     --model_args pretrained=meta-llama/Llama-2-7b-hf,pretrained_path={new_model_path} \
        --tasks hellaswag,swag,winogrande,boolq,xwinograd_en    --device cuda:0     --batch_size 4  --output_path logs/llama7b_2b-{layer_num}'
        os.system(command)

