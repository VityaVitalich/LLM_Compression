import torch
from pathlib import Path
from argparse import ArgumentParser
import os
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from peft import PeftModel 
from transformers import AutoModelForCausalLM, AutoTokenizer



if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument("--base_model", help="full model", required=True, type=str)
    parser.add_argument("--adapters_path", help="path to adapters saving dir", required=True, type=str)
    parser.add_argument("--save_dir", help="path to saving results", required=True, type=str)
    parser.add_argument("--token", help="Auth token", required=True, type=str)

    args = parser.parse_args()
    parent_directory = args.adapters_path
    save_path = args.save_dir
    prefix = 'checkpoint-'

    base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map='auto',
            use_auth_token=args.token)

    # Iterate over all items in the parent directory
    for item in os.listdir(parent_directory):
        # Construct the full path of the item
        item_path = os.path.join(parent_directory, item)
         # Check if the item is a directory and matches the pattern
        if os.path.isdir(item_path) and item.startswith(prefix):
            print(f"Found directory: {item}")
            # Here, you can add your code to process each "checkpoint-<number>" directory
            number_part = item[len(prefix):]
            checkpoint_num = int(number_part)


            model = PeftModel.from_pretrained(base_model, item_path) 
            merged_model = model.merge_and_unload() 
            
            tokenizer = AutoTokenizer.from_pretrained(item_path)
            result_path = f'{save_path}checkpoint-{number_part}/'
            merged_model.save_pretrained(result_path, from_pt=True)
            tokenizer.save_pretrained(result_path)

            del model
            del merged_model


