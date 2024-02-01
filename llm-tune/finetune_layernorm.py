from typing import Optional, Dict, Sequence

from dataclasses import dataclass, field
from itertools import chain

import numpy as np
import torch

import datasets
from datasets import load_dataset

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator
)

IGNORE_INDEX = -100

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

def load_hf_datasets(
    data_args
):
    # Load the dataset
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            streaming=data_args.streaming,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                streaming=data_args.streaming,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                streaming=data_args.streaming,
            )
        
        return raw_datasets

def tokenize_datasets(
    data_args,
    raw_datasets,
    tokenizer
):
    
    dataset_type = list(raw_datasets.keys())[0]
    column_names = list(raw_datasets[dataset_type].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        output = tokenizer(examples[text_column_name])
        return output
    
    if not data_args.streaming:
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    else:
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
        )

    return tokenized_datasets

def format_datasets(
    data_args,
    tokenized_datasets,
    tokenizer
):
    
    block_size = min(data_args.block_size, tokenizer.model_max_length)
    print(block_size)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(
        examples
    ):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        
        return result


    if not data_args.streaming:
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )
    else:
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
        )
    
    return lm_datasets

@dataclass
class DataCollatorWithMaskForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, batch):
        input_ids = []
        labels = []
        attention_masks = []

        for item_dict in batch:
            input_ids.append(torch.tensor(item_dict["input_ids"]))
            attention_masks.append(torch.tensor(item_dict["attention_mask"]))
            label = torch.tensor(item_dict["labels"])
            label[:-1] = IGNORE_INDEX
            labels.append(label)

        input_ids = torch.vstack(input_ids)
        attention_masks = torch.vstack(attention_masks)
        labels = torch.vstack(labels)
            
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict
    
def run_train(
    model_args,
    data_args,
    training_args
):
    
    # Load pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        token=model_args.token
    )

    # Load pretrained tokenizer
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)

    #Load and preprocessing dataset
    raw_datasets = load_hf_datasets(data_args)
    tokenized_datasets = tokenize_datasets(data_args, raw_datasets, tokenizer)
    lm_datasets = format_datasets(data_args, tokenized_datasets, tokenizer)

    data_collator = DataCollatorWithMaskForCausalLM(
        tokenizer=tokenizer
    )

    layernorm_names = [f"model.layers.{layer_block_num}.input_layernorm.weight" for layer_block_num in range(len(model.model.layers))]
    layernorm_names += [f"model.layers.{layer_block_num}.post_attention_layernorm.weight" for layer_block_num in range(len(model.model.layers))]

    #Set model parameters to be learned
    for name, param in model.named_parameters():
        if name not in layernorm_names:
            # freeze base model's layers
            param.requires_grad = False

    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"trainable_params: {trainable_params}")

    #Train
    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator
    )

    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload

def main():
    data_args = DataTrainingArguments(
        dataset_name = "wikitext",
        dataset_config_name = "wikitext-2-raw-v1",
        validation_split_percentage = 5,
        block_size = 128
    )

    model_args = ModelArguments(
        model_name_or_path = "/home/projects/Quantization/SpQR/weights/LLaMA7B_quant_threshold_063",
        config_name = "/home/projects/Quantization/SpQR/weights/LLaMA7B_quant_threshold_063/config.json",
        tokenizer_name = "/home/projects/Quantization/SpQR/weights/LLaMA7B_quant_threshold_063",
        use_fast_tokenizer = True,
        token = None,
        trust_remote_code = True
    )

    training_args = TrainingArguments(
        output_dir = "./exp_results/wikitext-2/layer_norm",
        overwrite_output_dir = True,
        learning_rate = 3e-4,
        seed = 11,
        num_train_epochs = 3,
        per_device_train_batch_size = 4,
        per_device_eval_batch_size = 4,
        gradient_accumulation_steps = 16,
        gradient_checkpointing=False,
        save_strategy = "epoch",
        weight_decay = 0.1,
        warmup_ratio = 0.03,
        lr_scheduler_type = "cosine",
        logging_steps = 1,
        do_train = True,
        do_eval = True,
        # report_to = "tensorboard"
        report_to = None
    )

    run_train(
        model_args,
        data_args,
        training_args
    )

if __name__ == "__main__":
    main()