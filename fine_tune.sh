#!/usr/bash
CONFIG_PATH='/home/LLM_Compression/configs/glm.py'

#python -m torch.distributed.launch --nproc_per_node 2 train.py --config_path=$CONFIG_PATH
python train.py --config_path=$CONFIG_PATH
