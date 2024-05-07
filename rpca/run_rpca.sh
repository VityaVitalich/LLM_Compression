MODEL='meta-llama/Llama-2-7b-hf'
TOKEN='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'
SAVE_DIR='/home/cache/Llama7b_rpca_no_r/'
RANK=1024

python run_rpca.py \
--model_path=$MODEL \
--token=$TOKEN \
--save_dir=$SAVE_DIR \
--rank=$RANK

