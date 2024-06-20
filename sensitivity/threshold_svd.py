import torch
from pathlib import Path
from argparse import ArgumentParser
import os
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

@torch.no_grad()
def apply_truncated_svd(tensor, n_components):
    shape = tensor.shape
    tensor_2d = tensor.detach().cpu().float().numpy()
    
    # Perform SVD using numpy
    U, S, Vt = np.linalg.svd(tensor_2d, full_matrices=False)
    # Keep only the top n_components
    U_reduced = U[:, :n_components]
    S_reduced = S[:n_components]
    Vt_reduced = Vt[:n_components, :]
    
    # Reconstruct the reduced tensor
    reduced_2d = (U_reduced @ np.diag(S_reduced)) @ Vt_reduced
    
    return torch.tensor(reduced_2d, dtype=torch.bfloat16)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model", help="path to model", required=True)
    parser.add_argument("--token", help="HF token", required=True)
    parser.add_argument("--n_components", help="number of components for SVD", required=True, type=int)
    parser.add_argument("--save_dir", help="Dir to save", required=True, type=str)

    args = parser.parse_args()
    layer_range = np.arange(0, 32)
    sa_projections = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    mlp_projections = ['gate_proj', 'up_proj', 'down_proj']

    logs_path = Path(args.save_dir)
    logs_path.mkdir(parents=True, exist_ok=True)

    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.model, use_auth_token=args.token, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_auth_token=args.token)
    
    thr = 0.76
    df = pd.read_csv('/home/LLM_Compression/logs/svd_eval/acc_norm_results.csv')
    df['layer_name'] = df['layer_name'].str.replace('svd-', '')
    df = df[df['acc_norm'] > thr]
    
    for i, row in df.iterrows():
        layer, number = row['layer_name'], row['layer_number']
        proj_name = layer + '_proj'
        number = int(number)
        print(proj_name, number)
        if proj_name in sa_projections:
            layer = getattr(model.model.layers[number].self_attn, proj_name)
        else:
            layer = getattr(model.model.layers[number].mlp, proj_name)
            layer.weight.data = apply_truncated_svd(layer.weight.data, args.n_components)
   
    new_model_name = f'Llama7b_svd_{thr}'
    new_model_path = logs_path / new_model_name
    model.save_pretrained(new_model_path)
    tokenizer.save_pretrained(new_model_path)


