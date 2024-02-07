from scipy.linalg import svdvals
import os
import pickle

SAVING_DIR='/home/data/taxonomy/'
os.environ["TRANSFORMERS_CACHE"] = SAVING_DIR + "hf_cache/"
os.environ["HF_HOME"] = SAVING_DIR + "hf_cache/"

import torch
from transformers import AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer, AutoConfig, PretrainedConfig
from peft import PeftModel 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':

    checkpoint = 'meta-llama/Llama-2-7b-hf'
    peft_model_id = "/home/data/compression/clip_sm_cache/fine_tuning/lora/checkpoint-174" 

    token = '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' 


    model = LlamaForCausalLM.from_pretrained(
                checkpoint,
                use_auth_token=token, 
                torch_dtype=torch.bfloat16).to('cuda')

    singulars = {}
    for name, param in model.named_parameters():
        if ('layers' in name) and ('mlp' in name) :
            print(name)
            matrix = param.detach().cpu().float().numpy()
            singular_values = svdvals(matrix)

            singulars[name] = singular_values

            with open('svds.pickle', 'wb') as f:
                pickle.dump(singulars, f)