from typing import Optional
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm
from dataclasses import dataclass, field
from functools import partial

import torch
from tqdm import tqdm
from peft import get_peft_model, TaskType, LoraConfig

from ste_utils import prepare_llama_ste, prepare_scales_quik
from collators import (
    DataCollatorWithMaskForCausalLM,
    DistillDataCollatorWithMaskForCausalLM,
)
from distill_trainer import DistillTrainer
from data_utils import (
    encode_with_messages_format,
    encode_with_prompt_completion_format,
    format_datasets,
    tokenize_datasets,
    load_hf_datasets,
)

# import transformers
from transformers_modified.src.transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
import transformers_modified.src.transformers as transformers


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
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
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
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
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
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
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

    dataset_percentage: Optional[int] = field(
        default=100,
        metadata={"help": "The number of percentage to take from entire dataset"},
    )
    seed: Optional[int] = field(
        default=42,
    )


@torch.no_grad()
def create_mask(weight, outlier_fraction):
    w = torch.clone(weight).float()
    w_flat = w.view(-1)
    lower_threshold, upper_threshold = (
        torch.kthvalue(
            w_flat,
            int(w_flat.numel() * outlier_fraction / 2),
        )[0],
        torch.kthvalue(
            w_flat,
            int(w_flat.numel() * (1 - outlier_fraction / 2)),
        )[0],
    )

    outliers = (w < lower_threshold) | (w > upper_threshold)

    return ~outliers.detach()


def make_zero_outliers(model, outlier_fraction):
    for name, param in tqdm(model.named_parameters()):
        if "layers" in name:
            mask = create_mask(param.data, outlier_fraction)
            param.data *= mask.to(param.data.device)


def run_train(
    model_args,
    data_args,
    training_args,
    config,
):
    # Load pretrained model
    # if config.model_type == 'Llama':
    #     model_type = LlamaForCausalLM
    # else:
    #     model_type = AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        token=model_args.token,
        cache_dir=model_args.cache_dir,
        device_map="auto",
    )

    if config.zero_outliers:
        make_zero_outliers(model, config.outlier_fraction)

    if config.use_clip_softmax:
        model.set_clipped_sm(
            gamma=config.clip_softmax_gamma, eta=config.clip_softmax_eta
        )

    if config.ste.enable:
        outlier_ids, layer_bit = prepare_llama_ste(
            config.ste.path_to_act_scales,
            config.ste.fp_features_num,
            **config.ste.layer_bits,
        )

        if config.ste.quik_scales_path is not None:
            quik_scales = prepare_scales_quik(config.ste.quik_scales_path)
        else:
            quik_scales = None

        model.enable_ste(
            outlier_ids=outlier_ids,
            layer_bit=layer_bit,
            block_size=config.ste.block_size,
            learnable_scales=config.ste.learnable_scales,
            quik_scales=quik_scales,
        )

    if config.use_lora:
        task_type = TaskType.CAUSAL_LM
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ]
        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            init_lora_weights=True,
        )
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
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs
        )

    if isinstance(tokenizer, transformers.LlamaTokenizer) or isinstance(
        tokenizer, transformers.LlamaTokenizerFast
    ):
        num_added_tokens = tokenizer.add_special_tokens(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
            }
        )
        assert num_added_tokens in [
            0,
            1,
        ], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
    else:
        if not tokenizer.pad_token_id:
            tokenizer.pad_token = tokenizer.eos_token

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    print(len(tokenizer), embedding_size)

    # Load and preprocessing dataset
    raw_datasets = load_hf_datasets(data_args)

    if not config.distillation:
        ### instruct
        if config.data.instruct:
            # Preprocessing the datasets.
            if (
                "prompt" in raw_datasets["train"].column_names
                and "completion" in raw_datasets["train"].column_names
            ):
                encode_function = partial(
                    encode_with_prompt_completion_format,
                    tokenizer=tokenizer,
                    max_seq_length=data_args.block_size,
                )
            elif "messages" in raw_datasets["train"].column_names:
                encode_function = partial(
                    encode_with_messages_format,
                    tokenizer=tokenizer,
                    max_seq_length=data_args.block_size,
                )

            lm_datasets = raw_datasets.map(
                encode_function,
                batched=False,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=[
                    name
                    for name in raw_datasets["train"].column_names
                    if name not in ["input_ids", "labels", "attention_mask"]
                ],
                desc="Tokenizing and reformatting instruction data",
            )

            lm_datasets.set_format(type="pt")
            lm_datasets = lm_datasets.filter(
                lambda example: (example["labels"] != -100).any()
            )

            data_collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer, model=model, padding="longest"
            )
        else:
            tokenized_datasets = tokenize_datasets(data_args, raw_datasets, tokenizer)
            lm_datasets = format_datasets(data_args, tokenized_datasets, tokenizer)

            data_collator = DataCollatorWithMaskForCausalLM(tokenizer=tokenizer)
    else:
        lm_datasets = raw_datasets
        print(lm_datasets)
        data_collator = DistillDataCollatorWithMaskForCausalLM(tokenizer=tokenizer)

    if config.norm_tweek:
        layernorm_names = [
            f"model.layers.{layer_block_num}.input_layernorm.weight"
            for layer_block_num in range(len(model.model.layers))
        ]
        layernorm_names += [
            f"model.layers.{layer_block_num}.post_attention_layernorm.weight"
            for layer_block_num in range(len(model.model.layers))
        ]

        # Set model parameters to be learned
        for name, param in model.named_parameters():
            if name not in layernorm_names:
                # freeze base model's layers
                param.requires_grad = False
            else:
                param.requires_grad = True

    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"trainable_params: {trainable_params}")

    # Train
    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # Initialize our Trainer
    if config.distillation:
        trainer = DistillTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            # Data collator will default to DataCollatorWithPadding, so we change it.
            data_collator=data_collator,
            temperature=config.temperature,
            lambda_param=config.lambda_param,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            # Data collator will default to DataCollatorWithPadding, so we change it.
            data_collator=data_collator,
        )

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
    parser.add_argument("--local-rank", type=int, default=-1)
    args = parser.parse_args()
    config = read_config(args.config_path, "model_configs")
    #  torch.cuda.set_device(args.local_rank)

    data_args = DataTrainingArguments(
        dataset_name=config.data.dataset_name,
        dataset_config_name=config.data.dataset_config_name,
        validation_split_percentage=config.data.valid_split,
        block_size=config.data.block_size,
        dataset_percentage=config.data.dataset_percentage,
        seed=config.seed
    )

    model_args = ModelArguments(
        model_name_or_path=config.model_name_or_path,  # "/home/projects/LLaMA/huggingface/Llama-2-7b-hf",
        config_name=config.model_config_name,  # "/home/projects/LLaMA/huggingface/Llama-2-7b-hf/config.json",
        tokenizer_name=config.tokenizer_name,  # "/home/projects/LLaMA/huggingface/Llama-2-7b-hf",
        use_fast_tokenizer=True,
        token=config.token,  # None,
        trust_remote_code=True,
        cache_dir=config.cache_dir,
    )

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=True,
        learning_rate=config.learning_rate,
        seed=config.seed,
        num_train_epochs=config.num_train_epochs,  # 3,
        per_device_train_batch_size=config.per_device_train_batch_size,  # 2,
        per_device_eval_batch_size=config.per_device_eval_batch_size,  # 2,
        gradient_accumulation_steps=config.gradient_accumulation_steps,  # 16,
        gradient_checkpointing=config.gradient_checkpointing,  # False,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        evaluation_strategy=config.evaluation_strategy,
        eval_steps=config.eval_steps,
        weight_decay=config.weight_decay,  # 0.1,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=1,
        do_train=True,
        do_eval=True,
        report_to=config.report_to,
        run_name=config.run_name,
        remove_unused_columns=(not config.distillation),  # False when distilling
    )

    run_train(model_args, data_args, training_args, config)


if __name__ == "__main__":
    main()
