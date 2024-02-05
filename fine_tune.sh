#!/usr/bash
CONFIG_PATH='/home/LLM_Compression/configs/llama.py'

python -m torch.distributed.launch --nproc_per_node 2 utils.py --config_path=$CONFIG_PATH
