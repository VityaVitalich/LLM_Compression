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

    logs_path = Path("./logs/")
    logs_path.mkdir(parents=True, exist_ok=True)

    args = parser.parse_args()
    model2 = torch.load(args.bit2_model, map_location='cuda')
    model4 = torch.load(args.bit4_model, map_location='cuda')

    for layer_num in tqdm(layer_range):
        new_model = deepcopy(model4)
        new_model.model.layers[layer_num].load_state_dict(model2.model.layers[layer_num].state_dict())

        model_name = Path(args.bit4_model).parts[-1]
        new_model_name = f'{model_name.split(".")[0]}_2b-layer{layer_num}.pt'
        new_model_path = Path(args.bit4_model).parent / new_model_name
        torch.save(new_model, new_model_path)
        del new_model
 #       command = f'lm_eval --model hf     --model_args pretrained=meta-llama/Llama-2-7b-hf,pretrained_path={new_model_path} \
  #      --tasks boolq    --device cuda:0     --batch_size 4  --output_path logs/{new_model_name}'
#        os.system(command)

