from argparse import ArgumentParser
from rpca import R_pca
import os
import pickle

#SAVING_DIR='/home/data/taxonomy/'
#os.environ["TRANSFORMERS_CACHE"] = SAVING_DIR + "hf_cache/"
#os.environ["HF_HOME"] = SAVING_DIR + "hf_cache/"

import torch
from transformers import AutoModelForCausalLM
import numpy as np
from rpca import R_pca

@torch.no_grad()
def decompose_and_back(M, rank):
    rpca = R_pca(M.float().numpy())
    L, S = rpca.fit(max_iter=10, iter_print=1, tol=1e-7)
    new = L
#    U, Sigma, V = np.linalg.svd(L)
#    new = U[:,:rank]@np.diag(Sigma[:rank])@V[:rank,:]

    return torch.tensor(new + S, dtype=torch.bfloat16)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_path", help="path to model", required=True)
    parser.add_argument("--token", help="hf token", required=True)
    parser.add_argument("--save_dir", help='directory to save', required=True)
    parser.add_argument("--rank", help='SVD rank', required=True, type=int)


    args = parser.parse_args()

    checkpoint = args.model_path
    token = args.token


    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        use_auth_token=token, 
        torch_dtype=torch.bfloat16, 
        device_map='cpu')

    for name, param in model.named_parameters():
        if ('layers' in name) and (('mlp' in name) or ('self_attn' in name)):
            print(name)
            new_param = decompose_and_back(param.data, args.rank)
            param.data = new_param
    model.save_pretrained(args.save_dir)
