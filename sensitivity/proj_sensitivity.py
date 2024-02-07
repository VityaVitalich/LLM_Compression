import torch
from pathlib import Path
from argparse import ArgumentParser
import os
import numpy as np
from copy import deepcopy
from tqdm import tqdm

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--bit4_model", help="path to 4bit model", required=True)
    parser.add_argument("--bit2_model", help="path to 2bit model", required=True)
   # parser.add_argument("--layer", help="layer to change", required=True, type=int)
    layer_range = np.arange(0, 32)
    sa_projections = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    mlp_projections = ['gate_proj', 'up_proj', 'down_proj']

    logs_path = Path("./logs/")
    logs_path.mkdir(parents=True, exist_ok=True)

    args = parser.parse_args()
    model2 = torch.load(args.bit2_model, map_location='cuda')
    model4 = torch.load(args.bit4_model, map_location='cuda')


    # SA
    for proj_name in tqdm(sa_projections):
      new_model = deepcopy(model4)
      for layer_num in layer_range:
          
        layer4 = getattr(new_model.model.layers[layer_num].self_attn, proj_name)
        layer2 = getattr(model2.model.layers[layer_num].self_attn, proj_name)
        layer4.load_state_dict(layer2.state_dict())

      model_name = Path(args.bit4_model).parts[-1]
      new_model_name = f'{model_name.split(".")[0]}_2b-{proj_name}.pt'
      new_model_path = Path(args.bit4_model).parent / new_model_name
      torch.save(new_model, new_model_path)
      del new_model
    
    # MLP
    for proj_name in tqdm(mlp_projections):
      new_model = deepcopy(model4)
      for layer_num in layer_range:
          
        layer4 = getattr(new_model.model.layers[layer_num].mlp, proj_name)
        layer2 = getattr(model2.model.layers[layer_num].mlp, proj_name)
        layer4.load_state_dict(layer2.state_dict())

      model_name = Path(args.bit4_model).parts[-1]
      new_model_name = f'{model_name.split(".")[0]}_2b-{proj_name}.pt'
      new_model_path = Path(args.bit4_model).parent / new_model_name
      torch.save(new_model, new_model_path)
      del new_model
    
