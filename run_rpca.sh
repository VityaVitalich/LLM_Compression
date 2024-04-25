MODEL='meta-llama/Llama-2-7b-hf'
TOKEN='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'
SAVE_DIR='/home/cache/rpca_test/'
RANK=128

python run_rpca.py \
--model_path=$MODEL \
--token=$TOKEN \
--save_dir=$SAVE_DIR \
--rank=$RANK

