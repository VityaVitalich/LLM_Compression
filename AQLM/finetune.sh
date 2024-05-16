MODEL_PATH='/home/cache/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9'
INPUT_PATH='/home/cache/aqlm_llama7_repoconf/'
DATASET_PATH='pajama'
SAVE_PATH='/home/cache/aqlm_llama7_repoconf_ft/'

python finetune.py \
  --base_model $MODEL_PATH \
  --quant_model $INPUT_PATH \
  --dataset $DATASET_PATH \
  --nsamples=2048 \
  --val_size=128 \
  --lr=1e-5 \
  --adam_beta1=0.90 \
  --adam_beta2=0.999 \
  --epochs=5 \
  --early_stop=3 \
  --batch_size=32 \
  --microbatch_size=4 \
  --save $SAVE_PATH \
  --gradient_checkpointing
