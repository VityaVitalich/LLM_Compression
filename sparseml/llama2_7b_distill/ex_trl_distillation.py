# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import DefaultDataCollator

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

from sparseml.transformers import (
    DataTrainingArguments,
    SparseAutoModelForCausalLM,
    SparseAutoTokenizer,
    TextGenerationDataset,
    TrainingArguments,
    Trainer
)

# from sft_trainer import SFTTrainer


model_path = "/home/llm_compression/Quantization/SparseGPT/output_llama7b_sparseml/stage_sparsity"
teacher_path = "/home/exp_results/instruct/llama_quik4bit3bit_normal_noise_wanda/merged_500"
output_dir = "/home/exp_results/kd/distill_llama_7b_ultrachat"

model = SparseAutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
)
teacher = SparseAutoModelForCausalLM.from_pretrained(
    teacher_path, torch_dtype="auto", device_map="auto"
)
tokenizer = SparseAutoTokenizer.from_pretrained(model_path)


# for name, param in model.named_parameters():
#     if name.find("model.layers.0") == -1:
#         param.requires_grad = False

# trainable_params = 0
# all_param = 0
# for _, param in model.named_parameters():
#     all_param += param.numel()
#     if param.requires_grad:
#         trainable_params += param.numel()

# print(f"trainable_params: {trainable_params}")

# model = AutoModelForCausalLM.from_pretrained(
#     model_path, torch_dtype="auto", device_map="auto"
# )
# teacher = AutoModelForCausalLM.from_pretrained(
#     teacher_path, torch_dtype="auto", device_map="auto"
# )
# tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load gsm8k using SparseML dataset tools
# data_args = DataTrainingArguments(
#     dataset="gsm8k", dataset_config_name="main", max_seq_length=32
# )

# Load ultrachat using SparseML dataset tools
data_args = DataTrainingArguments(
    dataset="ultrachat-200k", dataset_config_name="default", max_seq_length=1024
)

dataset_manager = TextGenerationDataset.load_from_registry(
    data_args.dataset,
    data_args=data_args,
    split="train",
    tokenizer=tokenizer,
)
train_dataset = dataset_manager.tokenize_and_process()
print(f"--> Training Set Length = {len(train_dataset)}")

# recipe for maintaining model sparsity during finetuning
# recipe = """
# test_stage:
#   pruning_modifiers:
#     ConstantPruningModifier:
#       targets: ['re:.*q_proj.weight', 're:.*k_proj.weight', 're:.*v_proj.weight',
#       're:.*o_proj.weight', 're:.*gate_proj.weight', 're:.*up_proj.weight',
#       're:.*down_proj.weight']
#       start: 0
#     OutputDistillationModifier:
#       targets: ['re:model.layers.\\d+$']
#       comparison: "square_head"
#       start: 0
#       orig_scale: 1.0
#       distill_scale: 1.0
# """

# recipe = """
# test_stage:
#   pruning_modifiers:
#     ConstantPruningModifier:
#       targets: ['re:.*q_proj.weight', 're:.*k_proj.weight', 're:.*v_proj.weight',
#       're:.*o_proj.weight', 're:.*gate_proj.weight', 're:.*up_proj.weight',
#       're:.*down_proj.weight']
#       start: 0
#     OutputDistillationModifier:
#       targets: ["model.layers.0"]
#       comparison: "square_head"
#       start: 0
#       orig_scale: 1.0
#       distill_scale: 1.0
# """

recipe = "/home/LLM_Compression/sparseml/llama2_7b_distill/configs/distill.yaml"

data_collator = DefaultDataCollator()

# training_args = TrainingArguments(
#     # run_name=config.run_name,
#     output_dir = config['output_dir'],
#     overwrite_output_dir = True,
#     learning_rate = config['learning_rate'], 
#     seed = config['seed'], 
#     max_steps = config['max_steps'],
#     # num_train_epochs = config.num_train_epochs, #3,
#     weight_decay = config['weight_decay'], #0.1,
#     warmup_ratio = config['warmup_ratio'],
#     lr_scheduler_type = config['lr_scheduler_type'],
#     per_device_train_batch_size = config['per_device_train_batch_size'], #2,
#     per_device_eval_batch_size = config['per_device_eval_batch_size'], #2,
#     gradient_accumulation_steps = config['gradient_accumulation_steps'], #16,
#     gradient_checkpointing=config['gradient_checkpointing'], #False,
#     save_strategy = config['save_strategy'],
#     save_steps = config['save_steps'],
#     # evaluation_strategy = config.evaluation_strategy,
#     # eval_steps = config.eval_steps,
#     logging_steps = 1,
#     do_train = True,
#     do_eval = True,
#     # report_to = config['report_to']
# )


training_args = TrainingArguments(
    recipe=recipe,
    output_dir=output_dir,
    num_train_epochs=0.1,
    logging_steps=50,
    gradient_checkpointing=True,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    lr_scheduler_type='cosine',
    per_device_train_batch_size = 4,
    save_strategy = 'steps',
    save_steps = 50,
    bf16=True,
    do_train=True,
    do_eval=False
)

# initialize trainer
# trainer = Trainer(
#     model=model,
#     # model_state_path=model_path,
#     recipe=training_args.recipe,
#     # recipe_args='{"num_epochs": 10.0, "qat_start_epoch": 7.0, "observer_epoch": 9.0}',
#     teacher=teacher,
#     # metadata_args=["per_device_train_batch_size","per_device_eval_batch_size","fp16"],
#     args=training_args,
#     train_dataset=train_dataset,
#     # eval_dataset=train_dataset["validation"],
#     tokenizer=tokenizer,

# )


trainer = Trainer(
    # model_init=model,
    model=model,
    # model_state_path=model_path,
    teacher=teacher,
    recipe=training_args.recipe,
    # recipe_args=training_args.recipe_args,
    args=training_args,
    data_args=data_args,
    train_dataset=train_dataset,
    eval_dataset=None,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
trainer.train()
trainer.save_model()