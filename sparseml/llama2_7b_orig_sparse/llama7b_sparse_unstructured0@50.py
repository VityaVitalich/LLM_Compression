import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainingArguments
)
from sparseml.transformers import SparseAutoModelForCausalLM, apply


# define a recipe to handle sparsity, finetuning and quantization
recipe = "/home/LLM_compression/sparseml/llama2_7b_orig_sparse/unstructured_0@50.yaml"

# load the model in as bfloat16 to save on memory and compute
model_name_or_path = "/home/LLaMA/huggingface/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map = 'auto'
)

# uses SparseML's built-in preprocessing for ultra chat
dataset = "ultrachat-200k"

# save location of quantized model
output_dir = "/home/exp_results/output_llama7b_sparseml"

# # set dataset config parameters
splits = {"calibration": "train_gen[:5%]", "train": "train_gen"}
max_seq_length = 512
num_calibration_samples = 512

# # set training parameters for finetuning
# num_train_epochs = 0.5
# logging_steps = 500
# save_steps = 5000
# gradient_checkpointing = True  # saves memory during training
# learning_rate = 0.0001
# bf16 = False  # using full precision for training
# lr_scheduler_type = "cosine"
# warmup_ratio = 0.1

# this will run the recipe stage by stage:
# oneshot sparsification

# accelerate launch \
#     --config_file example_fsdp_config.yaml \
#     --no_python sparseml.transformers.text_generation.oneshot \
#     --model PATH_TO_MODEL \
#     --dataset "gsm8k" \
#     --dataset_config_name "main" \
#     --concatenate_data OPTIONAL \
#     --recipe PATH_TO_RECIPE \
#     --output_dir PATH_TO_OUTPUT \
#     --splits "train" \
#     --pad_to_max_length False \
#     --oneshot_device DEVICE \
#     --num_calibration_samples 1024 \
#     --max_seq_len 4096


apply(
    model=model,
    dataset=dataset,
    recipe=recipe,
    output_dir=output_dir,
    splits=splits,
    max_seq_length=max_seq_length,
    num_calibration_samples=num_calibration_samples,
    pad_to_max_length=False
)
