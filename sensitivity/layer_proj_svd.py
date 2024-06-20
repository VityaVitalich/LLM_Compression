import torch
from pathlib import Path
from argparse import ArgumentParser
import os
import numpy as np
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

    # SA
    for layer_num in tqdm(layer_range):
        for proj_name in sa_projections:
            new_model = deepcopy(model)
            layer = getattr(new_model.model.layers[layer_num].self_attn, proj_name)
            layer.weight.data = apply_truncated_svd(layer.weight.data, args.n_components)

            new_model_name = f'Llama7b_svd-{proj_name}_layer-{layer_num}'
            new_model_path = logs_path / new_model_name
            new_model.save_pretrained(new_model_path)
            tokenizer.save_pretrained(new_model_path)
    # SA
        for proj_name in mlp_projections:
            new_model = deepcopy(model)
            layer = getattr(new_model.model.layers[layer_num].mlp, proj_name)
            layer.weight.data = apply_truncated_svd(layer.weight.data, args.n_components)

            new_model_name = f'Llama7b_svd-{proj_name}_layer-{layer_num}'
            new_model_path = logs_path / new_model_name
            new_model.save_pretrained(new_model_path)
            tokenizer.save_pretrained(new_model_path)
