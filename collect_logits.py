from argparse import ArgumentParser
import torch
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer
from train import DataTrainingArguments, ModelArguments
from data_utils import encode_with_prompt_completion_format, load_hf_datasets, tokenize_datasets, format_datasets, encode_with_messages_format


@torch.no_grad()
def format_logit(example):
    global model

    logits = model(torch.tensor(example['input_ids']).to(model.device), attention_mask=torch.tensor(example['attention_mask']).to(model.device)).logits
    example['logits'] = logits.cpu().tolist()
    return example

@torch.no_grad()
def format_logit_instruct(example):
    global model
    global tokenizer

    ids = torch.nn.utils.rnn.pad_sequence(
            example['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id
    ).to(model.device)
    masks =torch.nn.utils.rnn.pad_sequence(
            example['attention_mask'], batch_first=True, padding_value=0
    ).to(model.device)
    logits = model(ids, attention_mask=masks).logits
    new_ls = []
    for i, l in enumerate(logits):
        #print(e)
        mask = (ids[i] != tokenizer.pad_token_id)
        cur_logit = l[mask]
        new_ls.append(cur_logit.cpu().tolist())
    example['logits'] = new_ls
    return example

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_path", help="path to model", required=True)
    parser.add_argument("--dataset-name", help="dataset_name", required=True)
    parser.add_argument("--save_dir", help='directory to save', required=True)

    parser.add_argument("--dataset_config_name", help='dataset revision', required=False, default=None)
    parser.add_argument("--block_size", help='max len or block size', required=False, default=1024, type=int)
    parser.add_argument("--valid_split", help='validation size', required=False, default=10, type=int)
    parser.add_argument("--dataset_percentage", help='percentage of dataset', required=False, default=100, type=int)
    parser.add_argument("--cache_dir", help='cache directory', required=False, default="/home/data/taxonomy/hf_cache/")
    parser.add_argument("--token", help='HF token', required=False, default=None)
    parser.add_argument("--batch_size", help='batch_size', required=False, default=2, type=int)
    parser.add_argument("--instruct", help='instruction dataset', required=False, default=0, type=int)





    args = parser.parse_args()

    data_args = DataTrainingArguments(
        dataset_name = args.dataset_name,
        dataset_config_name = (None if args.instruct else args.dataset_config_name),
        validation_split_percentage = args.valid_split,
        block_size = args.block_size,
        dataset_percentage = args.dataset_percentage
    )

    model_args = ModelArguments(
        model_name_or_path = args.model_path,
        config_name = None, 
        tokenizer_name = None,
        use_fast_tokenizer = True,
        token = args.token,
        trust_remote_code = True,
        cache_dir=args.cache_dir
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

    tokenizer.pad_token = tokenizer.eos_token

    raw_datasets = load_hf_datasets(data_args)
    

    if args.instruct:
        logit_fn = format_logit_instruct

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
    else:
        logit_fn = format_logit

        tokenized_datasets = tokenize_datasets(data_args, raw_datasets, tokenizer)
        lm_datasets = format_datasets(data_args, tokenized_datasets, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        cache_dir=model_args.cache_dir,
        device_map="auto",
        token=model_args.token,	    
    )

    dataset_with_logits = lm_datasets.map(logit_fn, batched=True, batch_size=args.batch_size, desc=f"Obtaining logits")
    dataset_with_logits.save_to_disk(args.save_dir)
