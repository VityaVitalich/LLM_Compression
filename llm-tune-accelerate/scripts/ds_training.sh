accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes 1 \
    --use_deepspeed \
    --deepspeed_config_file /home/projects/Fine_tuning/fine_tune/ds_configs/stage3_offloading_accelerate.conf \
    /home/projects/Fine_tuning/fine_tune/finetune_instruct_with_accelerate.py --config_path=/home/projects/Fine_tuning/fine_tune/configs/llama_instruct.py