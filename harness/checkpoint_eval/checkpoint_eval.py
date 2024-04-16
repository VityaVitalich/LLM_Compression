import torch
from pathlib import Path
from argparse import ArgumentParser
import os
import numpy as np
from copy import deepcopy
from tqdm import tqdm

if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument("--model_dir", help="path to model", required=True, type=str)
    parser.add_argument("--save_dir", help="path to saving results", required=True, type=str)
    parser.add_argument("--task", default="paper", type=str) # paper or huawei
    parser.add_argument("--bs", default=16, type=int, help="batch size")
    args = parser.parse_args()
    parent_directory = args.model_dir
    save_path = args.save_dir
    prefix = 'checkpoint-'

    if args.task == 'paper':
        tasks = 'hellaswag,boolq,winogrande,piqa,ai2_arc'
    elif args.task == 'huawei':
        tasks = 'hellaswag,boolq,swag,winogrande,xwinograd_en'
    else:
        raise AttributeError(f"No such task {args.task}")
    # Iterate over all items in the parent directory
    for item in os.listdir(parent_directory):
        # Construct the full path of the item
        item_path = os.path.join(parent_directory, item)
         # Check if the item is a directory and matches the pattern
        print(item_path, item)
        if os.path.isdir(item_path) and item.startswith(prefix):
            print(f"Found directory: {item}")
            # Here, you can add your code to process each "checkpoint-<number>" directory
            number_part = item[len(prefix):]
            checkpoint_num = int(number_part)

        
            result_path = f'{save_path}_ckpt{checkpoint_num}/'
            command = f'lm_eval --model hf --model_args pretrained={item_path},parallelize=True \
                        --tasks {tasks} --batch_size {args.bs} --num_fewshot 0 --device cuda --output_path {result_path}'
            os.system(command)