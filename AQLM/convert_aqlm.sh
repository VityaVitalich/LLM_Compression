MODEL='/home/cache/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9'
IN_PATH='/home/cache/aqlm_llama7_repoconf/'
OUT_PATH='/home/cache/aqlm_llama7_repoconf_hf/'


python convert_to_hf.py \
--model $MODEL \
--in_path $IN_PATH \
--out_path $OUT_PATH \
--save_safetensors
