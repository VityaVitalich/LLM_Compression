# export CUDA_VISIBLE_DEVICES=0   # or e.g. 0,1,2,3
export MODEL_PATH='/home/cache/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9'
export DATASET_PATH='pajama' #'/home/data/compression/aqlm_cache/red_pajama_llama.pth'
export SAVE_PATH='/home/cache/aqlm_llama7_repoconf/'
export WANDB_PROJECT="AQLM"
export WANDB_NAME="TEST_AQLM"

python main.py $MODEL_PATH $DATASET_PATH \
 --nsamples=4096 \
 --val_size=128 \
 --num_codebooks=1 \
 --nbits_per_codebook=15 \
 --in_group_size=8 \
 --relative_mse_tolerance=0.01 \
 --finetune_lr=1e-4 \
 --finetune_adam_beta1=0.90 \
 --finetune_adam_beta2=0.95 \
 --finetune_batch_size=32 \
 --local_batch_size=2 \
 --finetune_max_epochs=50 \
 --finetune_keep_best \
 --offload_activations \
 --save $SAVE_PATH \
 --wandb 

