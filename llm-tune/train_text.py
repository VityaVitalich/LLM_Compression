from argparse import ArgumentParser
import logging
import math
import os
import random
import shutil
from pathlib import Path

from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from accelerate import Accelerator
from accelerate.checkpointing import save_accelerator_state
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainingArguments
)
from peft import (
    get_peft_model,
    TaskType,
    LoraConfig
)



from quant_utils import prepare_llama_quant

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
    max_memory: int = field(
        default=21,
        metadata={"help": "Free memory per gpu."}
    )
    lora_init: bool = field(
        default=False,
        metadata={"help": "True: Use zero and gaussian initialization; False: Load adapters from LoftQ in HF hub."},
    )
    rank: int = field(
        default=64,
        metadata={"help": "Rank of LoRA adapters. LoftQ does not require this config. Used for fp16 LoRA or QLoRA."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoftQ does not require this config. Used for QLoRA."},
    )
    quant_noise_config: dict = field(
        default=None,
        metadata={"help": "Parameters to add noise"},
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
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom dataset defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    max_seq_length: Optional[int] = field(
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
    dataset_percentage: Optional[int] = field(
        default=100,
        metadata={
            "help": "The percentage of the dataset used for computation"
        },  
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
            trust_remote_code=data_args.trust_remote_code
        )

        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                streaming=data_args.streaming,
                trust_remote_code=data_args.trust_remote_code
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                streaming=data_args.streaming,
                trust_remote_code=data_args.trust_remote_code
            )
        
        if data_args.dataset_percentage < 100:
            dataset_frac = data_args.dataset_percentage/100
            dataset_parts = raw_datasets['train'].train_test_split(train_size=dataset_frac)
            raw_datasets['train'] = dataset_parts['train']
            dataset_parts = raw_datasets['validation'].train_test_split(test_size=dataset_frac)
            raw_datasets['validation'] = dataset_parts['test']

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
    
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    print(max_seq_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(
        examples
    ):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
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
            desc=f"Grouping texts in chunks of {max_seq_length}",
        )
    else:
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
        )
    
    return lm_datasets

def read_config(conf_path, func_name: str):
    if isinstance(conf_path, str):
        conf_path = Path(conf_path)

    source = conf_path.read_text()
    bytecode = compile(source, conf_path.as_posix(), "exec")
    namespace = {
        "__file__": conf_path.as_posix(),
    }
    exec(bytecode, namespace)
    return namespace[func_name]()  # type: ignore

def run_train(
    config
):
    
    config_dict = dict(config)
    config_dict['data'] = dict(config_dict['data'])
    config_dict['quant_noise_config'] = dict(config_dict['quant_noise_config'])
    config_dict['outliers'] = dict(config_dict['outliers'])
    config_dict['QuantizedLinear'] = dict(config_dict['QuantizedLinear'])
    config_dict['NoiseQuant'] = dict(config_dict['NoiseQuant'])
    config = config_dict

    data_args = DataTrainingArguments(
        dataset_name = config['data']['dataset_name'],
        dataset_config_name = config['data']['dataset_config_name'],
        validation_split_percentage = config['data']['validation_split_percentage'],
        max_seq_length = config['data']['max_seq_length'],
        dataset_percentage = config['data']['dataset_percentage'],
        trust_remote_code = config['data']['trust_remote_code'],
        preprocessing_num_workers = config['data']['preprocessing_num_workers']
    )

    model_args = ModelArguments(
        model_name_or_path = config['model_name_or_path'], #"/home/projects/LLaMA/huggingface/Llama-2-7b-hf",
        config_name = config['model_config_name'], #"/home/projects/LLaMA/huggingface/Llama-2-7b-hf/config.json",
        tokenizer_name = config['tokenizer_name'], #"/home/projects/LLaMA/huggingface/Llama-2-7b-hf",
        use_fast_tokenizer = config['use_fast_tokenizer'],
        token = config['token'], #None,
        trust_remote_code = config['trust_remote_code'],
        max_memory = config['max_memory'],
        # cache_dir= config.cache_dir,
        rank = config['lora_rank'],
        lora_alpha = config['lora_alpha'],
        # quant_noise_config = config['quant_noise_config']
    )

    training_args = TrainingArguments(
        # run_name=config.run_name,
        output_dir = config['output_dir'],
        overwrite_output_dir = True,
        learning_rate = config['learning_rate'], 
        seed = config['seed'], 
        max_steps = config['max_steps'],
        # num_train_epochs = config.num_train_epochs, #3,
        weight_decay = config['weight_decay'], #0.1,
        warmup_ratio = config['warmup_ratio'],
        lr_scheduler_type = config['lr_scheduler_type'],
        per_device_train_batch_size = config['per_device_train_batch_size'], #2,
        per_device_eval_batch_size = config['per_device_eval_batch_size'], #2,
        gradient_accumulation_steps = config['gradient_accumulation_steps'], #16,
        gradient_checkpointing=config['gradient_checkpointing'], #False,
        save_strategy = config['save_strategy'],
        save_steps = config['save_steps'],
        # evaluation_strategy = config.evaluation_strategy,
        # eval_steps = config.eval_steps,
        logging_steps = 1,
        do_train = True,
        do_eval = True,
        # report_to = config['report_to']
    )
    
    # If limit on cuda memory is specified enforce the limit
    if model_args.max_memory > 0:
        mem_info = torch.cuda.mem_get_info()
        print("Memory info: \n{}".format(mem_info))
        # total_memory = mem_info[1] * 1e-9 # convert Bytes to GB
        total_memory = torch.cuda.get_device_properties(0).total_memory  * 2**(-30) # convert Bytes to GB
        if model_args.max_memory > total_memory:
            raise ValueError("The specified memory limit {} is greater than the available memory {}.".format(model_args.max_memory, total_memory))
        else:
            fraction = model_args.max_memory / total_memory
            torch.cuda.set_per_process_memory_fraction(fraction)
            print("Restricting the memory to {} of the total memory to have a limit of {} ({} x {})".format(fraction, model_args.max_memory, fraction, total_memory))

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

    # Load pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        token=model_args.token,
        device_map = 'auto'
    )

    for name, param in model.named_parameters():
        param.requires_grad = False

    # if config.use_clip_softmax:
    #     model.set_clipped_sm(gamma=config.clip_softmax_gamma, eta=config.clip_softmax_eta)

    # if config.ste.enable:
    #     outlier_ids, layer_bit = prepare_llama_quant(config.ste.path_to_act_scales, config.ste.fp_features_num, **config.ste.layer_bits)
    #     model.enable_ste(outlier_ids=outlier_ids, layer_bit=layer_bit, block_size=config.ste.block_size)

    if config['QuantizedLinear']['replace']:
        outliers_config= config['outliers']
        outlier_ids, layer_bit = prepare_llama_quant(
            outliers_config['path_to_act_scales'], 
            outliers_config['fp_features_num']
        )

        model.replace_Linear(
            outlier_ids=outlier_ids,
            training_mode=config['QuantizedLinear']['training_mode'] 
        )

    if config['NoiseQuant']['add_quant_noise']:
        noise_config = config['NoiseQuant']
        outliers_config= config['outliers']
        outlier_ids, layer_bit = prepare_llama_quant(
            outliers_config['path_to_act_scales'], 
            outliers_config['fp_features_num'], 
            **noise_config['layer_bits']
        )
        model.add_quant_noise_to_weight( 
            layer_bit=layer_bit, 
            block_size=noise_config['block_size'],
            fp_cols_num=outliers_config['fp_features_num'],
            compute_scale=noise_config['compute_scale'], 
            quant_noise_predict=noise_config['predict']
        )

    if config['use_lora']:
        task_type = TaskType.CAUSAL_LM
        target_modules = config['lora_target_modules']
        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=model_args.rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
            init_lora_weights=True,
            # quant_noise_config=model_args.quant_noise_config
        )
        model = get_peft_model(model, lora_config)


    #Load and preprocessing dataset

    tokenizer.pad_token = tokenizer.eos_token
    raw_datasets = load_hf_datasets(data_args)
    tokenized_datasets = tokenize_datasets(data_args, raw_datasets, tokenizer)
    lm_datasets = format_datasets(data_args, tokenized_datasets, tokenizer)

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    print("dataset prepared")
    # data_collator = DataCollatorWithMaskForCausalLM(
    #     tokenizer=tokenizer
    # )

    if config['norm_tweek']:
        names_of_layernorm_layers = []
        module_name_dict = {name: module for name, module in model.named_modules()}
        for name, module in module_name_dict.items():
            if isinstance(module, transformers.models.llama.modeling_llama.LlamaRMSNorm):
                names_of_layernorm_layers.append(name)

        for name, param in model.named_parameters():
            name = name.replace('.weight', '')
            if name in names_of_layernorm_layers:
                param.requires_grad_()

    if config['train_lm_head']:
        for name, param in model.named_parameters():
            name = name.replace('.weight', '')
            if name == 'lm_head':
                param.requires_grad_()

    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"trainable_params: {trainable_params}")

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")
    )

    if config['resume_from_checkpoint'] is not None:
        print('resume')
        print(training_args.resume_from_checkpoint)
        train_result = trainer.train(resume_from_checkpoint=True)
    else:
        train_result = trainer.train()

    trainer.save_model()  # Saves the tokenizer too for easy upload


def main():
    parser = ArgumentParser()
    parser.add_argument("--config_path", help="path_to_conifg", required=True)

    args = parser.parse_args()
    config = read_config(args.config_path, 'model_configs')

    run_train(
        config
    )

if __name__ == "__main__":
    main()