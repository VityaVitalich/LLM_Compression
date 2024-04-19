export CUDA_VISIBLE_DEVICES=0   # or e.g. 0,1,2,3
export MODEL_PATH='/home/data/compression/Llama-2-7b-hf'
export DATASET_PATH='wikitext2' #'/home/data/compression/aqlm_cache/red_pajama_llama.pth'
export SAVE_PATH='/home/data/compression/aqlm_cache/llama7/'
#export WANDB_PROJECT=MY_AQ_EXPS
#export WANDB_NAME=COOL_EXP_NAME

python main.py $MODEL_PATH $DATASET_PATH \
 --nsamples=1024 \
 --val_size=128 \
 --num_codebooks=1 \
 --nbits_per_codebook=16 \
 --in_group_size=8 \
 --relative_mse_tolerance=0.01 \
 --finetune_batch_size=32 \
 --finetune_max_epochs=10 \
 --finetune_early_stop=3 \
 --finetune_keep_best \
 --local_batch_size=1 \
 --offload_activations \
 --save $SAVE_PATH
# --wandb \
# --resume \

