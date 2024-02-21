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


class LinearQuantNoise(nn.Module):
    def __init__(
        self, 
        weight, 
        bias, 
        quant_bit, 
        block_size, 
        train_quant_scales=False,
        fp_cols_inds=None, 
        training_mode: str = None, 
        add_quant_noise: bool =False
    )-> None:

        super().__init__()

        self.weight = nn.Parameter(weight.data)
        if bias is not None:
            self.bias = nn.Parameter(bias.data)
        else:
            self.bias = None

        self.fp_cols_inds = fp_cols_inds
        self.quant_bit = quant_bit
        self.quant_block_size = block_size
        self.quant_scales = None

        self.train_quant_scales = train_quant_scales
        self.delta_quant_scales = None

        self.add_quant_noise = add_quant_noise

        self.fp_weight_mask = None
        self.training_mode = training_mode

    def get_fp_weight_mask(self) -> None:
        with torch.no_grad():
            assert self.fp_cols_inds is not None, 'columns with fp weight are not given!'

            in_features = self.weight.shape[1]

            self.fp_weight_mask = [
                True if idx in self.fp_cols_inds else False \
                for idx in range(in_features)
            ]
            self.fp_weight_mask = torch.tensor(self.fp_weight_mask)

    def get_quant_scales(self) -> None:
        if self.fp_weight_mask is None:
            self.get_fp_weight_mask()

        w = self.weight.clone().detach()
        w_mask = ~self.fp_weight_mask
        in_features = self.weight.shape[1]
        block_size = self.quant_block_size
        quant_bit = self.quant_bit
        
        w_mask = w_mask.to(w.device)
        w = w_mask * w

        if (self.quant_block_size == 0):
            scales = torch.max(w, axis=0)[0]
        else:
            scales = torch.ones(in_features, dtype=w.dtype, device=w.device)
            for i in range(0, in_features, block_size):
                block_scale = torch.max(w[:, i:(i + block_size)])
                scales[i:(i + block_size)] = block_scale * scales[i:(i + block_size)]

        scales = scales / (2**quant_bit - 1)

        if self.fp_cols_inds is not None:
            scales[self.fp_cols_inds] = 0.0

        self.quant_scales = scales

    def quant_noise(self, w) -> None:
        w_rand = torch.randn_like(w, requires_grad=False) / 2
        w_rand = self.quant_scales * w_rand
        return w_rand

    def forward(self, x):
        device = self.weight.device

        if self.fp_weight_mask.device != device:
            self.fp_weight_mask = self.fp_weight_mask.to(device)     

        if self.training_mode == 'train_fp_weight':
            int_weight = self.weight.clone().detach()
            w = torch.where(self.fp_weight_mask, self.weight, int_weight)
        elif self.training_mode == 'train_int_weight':
            fp_weight = self.weight.clone().detach()
            w = torch.where(self.fp_weight_mask, fp_weight, self.weight)

        if self.add_quant_noise:
            if self.quant_scales.device != device:
                self.quant_scales = self.quant_scales.to(device)

            w_rand = self.quant_noise(w)
            w = w + w_rand

        output = F.linear(x, w, self.bias)
        return output

    def extra_repr(self) -> str:
        in_features = self.weight.shape[1]
        out_features = self.weight.shape[0]
        return f'in_features={in_features}, out_features={out_features}, bias={self.bias is not None}'

def get_fp_inds_for_quik(path_to_act_scales, fp_features_num):
    act_scales = torch.load(path_to_act_scales)
    fp_indices_in_lin_layers = {k: torch.sort(v)[1][-fp_features_num:] for k, v in act_scales.items()}
    return fp_indices_in_lin_layers

def encode_with_prompt_completion_format(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated 
    and it doesn't make sense to follow directly with the completion.
    '''
    # if prompt doesn't end with space and completion doesn't start with space, add space
    if not example['prompt'].endswith((' ', '\n', '\t')) and not example['completion'].startswith((' ', '\n', '\t')):
        example_text = example['prompt'] + ' ' + example['completion']
    else:
        example_text = example['prompt'] + example['completion']
    example_text = example_text + tokenizer.eos_token
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(example['prompt'], return_tensors='pt', max_length=max_seq_length, truncation=True)
    # mask the prompt part for avoiding loss
    labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }

def encode_with_messages_format(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    '''
    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')
    
    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text
        
    example_text = _concat_messages(messages).strip()
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]), return_tensors='pt', max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx+1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = _concat_messages(messages[:message_idx+1]) + "<|assistant|>\n"
            else:
                messages_so_far = _concat_messages(messages[:message_idx+1])
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt', 
                max_length=max_seq_length, 
                truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100
            
            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }

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
    config_dict['LinearQuantNoise'] = dict(config_dict['LinearQuantNoise'])
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
        quant_noise_config = config['quant_noise_config']
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
    if config['use_lora']:
        model = get_peft_model(model, lora_config)


    #Load and preprocessing dataset

    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        })
        assert num_added_tokens in [0, 1], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    print(len(tokenizer), embedding_size)

    raw_datasets = load_hf_datasets(data_args)

    # Preprocessing the datasets.
    if "prompt" in raw_datasets["train"].column_names and "completion" in raw_datasets["train"].column_names:
        encode_function = partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length=data_args.max_seq_length,
        )
    elif "messages" in raw_datasets["train"].column_names:
        encode_function = partial(
            encode_with_messages_format,
            tokenizer=tokenizer,
            max_seq_length=data_args.max_seq_length,
        )

    lm_datasets = raw_datasets.map(
        encode_function,
        batched=False,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=[name for name in raw_datasets["train"].column_names if name not in ["input_ids", "labels", "attention_mask"]],
        desc="Tokenizing and reformatting instruction data",
    )

    lm_datasets.set_format(type="pt")
    lm_datasets = lm_datasets.filter(lambda example: (example['labels'] != -100).any())

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    print("dataset prepared")
    # data_collator = DataCollatorWithMaskForCausalLM(
    #     tokenizer=tokenizer
    # )
    for name, param in model.named_parameters():
        param.requires_grad = False

    if config['LinearQuantNoise']['replace_Linear']:
        if (config['LinearQuantNoise']['path_to_act_scales'] is not None) and \
            (config['LinearQuantNoise']['fp_features_num'] is not None):
            path_to_act_scales = config['LinearQuantNoise']['path_to_act_scales']
            fp_features_num = config['LinearQuantNoise']['fp_features_num']
            fp_inds_in_lin_layers = get_fp_inds_for_quik(path_to_act_scales, fp_features_num)

        assert config['LinearQuantNoise']['quant_target_modules'] is not None, 'quant_target_modules shoud be given!'
        modules_name_dict = {name: module for name, module in model.named_modules()}
        for name, module in modules_name_dict.items():
            if isinstance(module, nn.Linear) and (name.find('lm_head') == -1):
                ind = name.rfind(".")
                if ind == -1:
                    father = modules_name_dict[""]
                else:
                    father = modules_name_dict[name[:ind]]
                print(name)
                fp_cols_inds = fp_inds_in_lin_layers[name]
                qlinear = LinearQuantNoise(
                    module.weight, module.bias, 
                    quant_bit=config['LinearQuantNoise']['quant_bit'], 
                    block_size=config['LinearQuantNoise']['block_size'], 
                    fp_cols_inds=fp_cols_inds, 
                    training_mode=config['LinearQuantNoise']['training_mode'], 
                    add_quant_noise=config['LinearQuantNoise']['add_quant_noise']
                )
                qlinear.get_quant_scales()
                setattr(father, name[ind + 1:], qlinear)

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