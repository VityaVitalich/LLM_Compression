import os
from typing import Optional, Dict, Sequence
from argparse import ArgumentParser
from pathlib import Path
import copy

from dataclasses import dataclass, field
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import datasets
from datasets import load_dataset, Dataset
import pandas as pd

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
    DataCollatorForLanguageModeling,
    Seq2SeqTrainer
)

from peft import PeftModel, get_peft_model, TaskType, LoraConfig

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

    eval_dataset_size: int = field(
        default=1024, metadata={"help": "Size of validation dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    dataset: str = field(
        default='alpaca',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
    dataset_format: Optional[str] = field(
        default=None,
        metadata={"help": "Which dataset format is used. [alpaca|chip2|self-instruct|hh-rlhf]"}
    )


@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if False:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict


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

def extract_unnatural_instructions_data(examples, extract_reformulations=False):
    out = {
        'input': [],
        'output': [],
    }
    for example_instances in examples['instances']:
        for instance in example_instances:
            out['input'].append(instance['instruction_with_input'])
            out['output'].append(instance['output'])
    if extract_reformulations:
        for example_reformulations in examples['reformulations']:
            if example_reformulations is not None:
                for instance in example_reformulations:
                    out['input'].append(instance['instruction_with_input'])
                    out['output'].append(instance['output'])
    return out

ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}

def extract_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {'input': prompt_format.format(**example)}

def local_dataset(dataset_name):
    if dataset_name.endswith('.json') or dataset_name.endswith('.jsonl'):
        full_dataset = Dataset.from_json(path_or_paths=dataset_name)
    elif dataset_name.endswith('.csv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))
    elif dataset_name.endswith('.tsv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name, delimiter='\t'))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")

    split_dataset = full_dataset.train_test_split(test_size=0.1)
    return split_dataset

def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }

    Available datasets to be selected with `dataset` argument:
        - alpaca, 52002 examples
        - alpaca cleaned, 51942 examples
        - chip2 (OIG), 210289 examples
        - self-instruct, 82612 examples
        - hh-rlhf (Anthropic), 160800 examples
        - longform, 23.7k examples
        - oasst1 (OpenAssistant) primary message tree only, 9,846 examples

    Coming soon:
        - unnatural instructions core, 66010 examples
        - unnatural instructions full, 240670 examples
        - alpaca-gpt4, 52002 examples
        - unnatural-instructions-gpt4, 9000 examples
        - supernatural-instructions, 69624 examples (same as paper with 100 ex/task more can be used)
        - flan (FLAN v2), up to 20M examples available
        - vicuna

    """
    def load_data(dataset_name):
        if dataset_name == 'alpaca':
            return load_dataset("tatsu-lab/alpaca")
        elif dataset_name == 'alpaca-clean':
            return load_dataset("yahma/alpaca-cleaned")
        elif dataset_name == 'chip2':
            return load_dataset("laion/OIG", data_files='unified_chip2.jsonl')
        elif dataset_name == 'self-instruct':
            return load_dataset("yizhongw/self_instruct", name='self_instruct')
        elif dataset_name == 'hh-rlhf':
            return load_dataset("Anthropic/hh-rlhf")
        elif dataset_name == 'longform':
            return load_dataset("akoksal/LongForm")
        elif dataset_name == 'oasst1':
            return load_dataset("timdettmers/openassistant-guanaco")
        elif dataset_name == 'vicuna':
            raise NotImplementedError("Vicuna data was not released.")
        else:
            if os.path.exists(dataset_name):
                try:
                    args.dataset_format = args.dataset_format if args.dataset_format else "input-output"
                    full_dataset = local_dataset(dataset_name)
                    return full_dataset
                except:
                    raise ValueError(f"Error loading dataset from {dataset_name}")
            else:
                raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")

    def format_dataset(dataset, dataset_format):
        if (
            dataset_format == 'alpaca' or dataset_format == 'alpaca-clean' or
            (dataset_format is None and args.dataset in ['alpaca', 'alpaca-clean'])
        ):
            dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
        elif dataset_format == 'chip2' or (dataset_format is None and args.dataset == 'chip2'):
            dataset = dataset.map(lambda x: {
                'input': x['text'].split('\n<bot>: ')[0].replace('<human>: ', ''),
                'output': x['text'].split('\n<bot>: ')[1],
            })
        elif dataset_format == 'self-instruct' or (dataset_format is None and args.dataset == 'self-instruct'):
            for old, new in [["prompt", "input"], ["completion", "output"]]:
                dataset = dataset.rename_column(old, new)
        elif dataset_format == 'hh-rlhf' or (dataset_format is None and args.dataset == 'hh-rlhf'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['chosen']
            })
        elif dataset_format == 'oasst1' or (dataset_format is None and args.dataset == 'oasst1'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['text'],
            })
        elif dataset_format == 'input-output':
            # leave as is
            pass
        # Remove unused columns.
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names['train'] if col not in ['input', 'output']]
        )
        return dataset

     # Load dataset.
    dataset = load_data(args.dataset)
    dataset = format_dataset(dataset, args.dataset_format)

    # Split train/eval, reduce size
    # if args.do_eval or args.do_predict:
    #     if 'eval' in dataset:
    #         eval_dataset = dataset['eval']
    #     else:
    #         print('Splitting train dataset in train and validation according to `eval_dataset_size`')
    #         dataset = dataset["train"].train_test_split(
    #             test_size=args.eval_dataset_size, shuffle=True, seed=42
    #         )
    #         eval_dataset = dataset['test']
    #     if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
    #         eval_dataset = eval_dataset.select(range(args.max_eval_samples))
    #     if args.group_by_length:
    #         eval_dataset = eval_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

    train_dataset = dataset['train']
    if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
        train_dataset = train_dataset.select(range(args.max_train_samples))
    if True:
        train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=False,
        predict_with_generate=False,
    )
    return dict(
        train_dataset=train_dataset if True else None,
        eval_dataset=eval_dataset if False else None,
        predict_dataset=eval_dataset if False else None,
        data_collator=data_collator
    )

    
def run_train(
    config
):
    
    config_dict = dict(config)
    config_dict['data'] = dict(config_dict['data'])
    config_dict['quant_noise_config'] = dict(config_dict['quant_noise_config'])
    config_dict['LinearQuantNoise'] = dict(config_dict['LinearQuantNoise'])
    config = config_dict

    data_args = DataTrainingArguments(
        dataset = config['data']['dataset'],
        dataset_format= config['data']['dataset_format'],
        source_max_len = config['data']['source_max_len'],
        target_max_len = config['data']['target_max_len']
    )

    model_args = ModelArguments(
        model_name_or_path = config['model_name_or_path'], #"/home/projects/LLaMA/huggingface/Llama-2-7b-hf",
        config_name = config['model_config_name'], #"/home/projects/LLaMA/huggingface/Llama-2-7b-hf/config.json",
        tokenizer_name = config['tokenizer_name'], #"/home/projects/LLaMA/huggingface/Llama-2-7b-hf",
        use_fast_tokenizer = config['use_fast_tokenizer'],
        token = config['token'], #None,
        trust_remote_code = config['trust_remote_code'],
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
        report_to = config['report_to']
    )
    
    task_type = TaskType.CAUSAL_LM
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    lora_config = LoraConfig(
        task_type=task_type,
        inference_mode=False,
        r=model_args.rank,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=0.1,
        target_modules=target_modules,
        init_lora_weights=True,
        quant_noise_config=model_args.quant_noise_config
    )
    
    # Load pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        token=model_args.token,
        device_map = 'auto'
    )
    if config['use_lora']:
        model = get_peft_model(model, lora_config)

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
    tokenizer.pad_token = tokenizer.eos_token


    #Load and preprocessing dataset
    data_module = make_data_module(tokenizer=tokenizer, args=data_args)

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
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
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
