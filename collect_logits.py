from argparse import ArgumentParser
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from train import DataTrainingArguments, ModelArguments, load_hf_datasets, tokenize_datasets, format_datasets


def format_logit(example):
    global model

    logits = model(torch.tensor(example['input_ids']).to(model.device), attention_mask=torch.tensor(example['attention_mask']).to(model.device)).logits
    example['logits'] = logits.cpu().tolist()
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




    args = parser.parse_args()

    data_args = DataTrainingArguments(
        dataset_name = args.dataset_name,
        dataset_config_name = args.dataset_config_name,
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

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        cache_dir=model_args.cache_dir,
        device_map="auto"	    
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

    raw_datasets = load_hf_datasets(data_args)
    tokenized_datasets = tokenize_datasets(data_args, raw_datasets, tokenizer)
    lm_datasets = format_datasets(data_args, tokenized_datasets, tokenizer)

    dataset_with_logits = lm_datasets.map(format_logit, batched=True, batch_size=args.batch_size, desc=f"Obtaining logits")
    dataset_with_logits.save_to_disk(args.save_dir)
