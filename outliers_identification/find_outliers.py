import os
import math
import functools
from functools import partial

from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

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
from datasets import load_dataset


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
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
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

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_path: Optional[str] = field(
        default=None, metadata={"help": "The path to dataset which will use for outliers analysis."}
    )
    output_path: Optional[str] = field(
        default=None, metadata={"help": "The path to file where outliers will be saved."}
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
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    num_samples: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "Number of samples which will be used to find  outliers. "
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

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


def get_model_and_tokenizer(model_args):
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map = 'auto'
    )

    return model, tokenizer

def get_sq_act_scales(model, tokenizer, dataset_path, num_samples=512, seq_len=512):
    model.eval()
    # device = next(model.parameters()).device
    device = model.device
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(functools.partial(stat_input_hook, name=name))
            )

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.shuffle(seed=42)

    for i in tqdm(range(num_samples)):
        input_ids = tokenizer(
            dataset[i]["text"], return_tensors="pt", max_length=seq_len, truncation=True
        ).input_ids.to(device)
        model(input_ids)

    for h in hooks:
        h.remove()

    return act_scales

class SymQuant:
    def __init__(
        self,
        out_features,
        bit
    ):

        self.bit = torch.tensor(bit)
        self.alpha_scale = torch.zeros((out_features, 1))

        self.qmin = None
        self.qmax = None
    
    def compute_alpha_scale(self, quant_weight) -> None:
        w = quant_weight.data
        device = quant_weight.device
        alpha = self.alpha_scale
        bit = self.bit

        if alpha.device != device:
            alpha = alpha.to(device)
        if bit.device != device:
            bit = bit.to(device)

        out_features = w.shape[0]
        
        alpha, qmax, qmin = self._get_row_scale(w, bit)
        alpha = alpha.to(w.dtype)
        self.alpha_scale.data = alpha.reshape((out_features, 1))
        self.qmax = qmax
        self.qmin = qmin

    def _get_row_scale(self, w, bit, maxshrink=0.8, grid=100, norm=2):
        qmax = 2 ** (bit.detach() - 1) - 1
        qmin = -(2 ** (bit.detach() - 1))
        tmp = torch.zeros(w.shape[0], device=w.device)
        best = torch.full([w.shape[0]], float('inf'), device=w.device)

        wmin = torch.minimum(w.min(1)[0], tmp)
        wmax = torch.maximum(w.max(1)[0], tmp)

        wmax = torch.maximum(torch.abs(wmin), wmax)
        tmp = wmin < 0
        if torch.any(tmp):
            wmin[tmp] = -wmax[tmp]

        tmp = (wmax == 0)
        wmax[tmp] = +1

        alpha = wmax

        for i in range(int(maxshrink * grid)):
            p = 1 - i / grid 
            wmax1 = p * wmax

            delta1 = wmax1 / qmax

            #quantization
            q = torch.clamp(torch.round(w / delta1.unsqueeze(1)), qmin, qmax)
            #dequantization
            q = q * delta1.unsqueeze(1)

            q -= w
            q.abs_()
            q.pow_(norm)
            err = torch.sum(q, 1)
            tmp = err < best

            if torch.any(tmp):
                best[tmp] = err[tmp]
                alpha[tmp] = wmax1[tmp]

        return alpha, qmax, qmin
    

    def quantize(self, weight, dequantize=True):
        if torch.cuda.is_available:
            w = weight.to('cuda')
            alpha = self.alpha_scale.to('cuda')
            qmax = self.qmax.to('cuda')
            qmin = self.qmin.to('cuda')
        else:
            w = weight
            alpha = self.alpha_scale
            qmax = self.qmax
            qmin = self.qmin

        alpha = alpha.flatten()
        delta = alpha / self.qmax
        q = torch.round(w / delta)
        q = torch.clamp(q, qmin, qmax)

        if dequantize:
            q = delta * q 

        return q

class OBS_estimator:
    def __init__(
        self,
        name,
        ncolumns,
        device
    ):
        self.name = name
        self.ncolumns = ncolumns
        self.device = device
        self.percdamp = .01
        self.H = torch.zeros((self.ncolumns, self.ncolumns), device='cpu')
        self.nsamples = 0
        self.quantizer = None

    def add_sym_quantizer(self, weight, bit):
        self.quantizer = SymQuant(out_features=self.ncolumns, bit=bit)
        self.quantizer.compute_alpha_scale(weight)

    def collect_stat(self, inp):
        # if len(inp.shape) == 2:
        #     inp = inp.unsqueeze(0)
        
        # if len(inp.shape) == 3:
        #     inp = inp.reshape((-1, inp.shape[-1]))
        
        inp = inp.reshape((-1, inp.shape[-1]))
        tmp = inp.shape[0]    
        inp = inp.t()

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = np.sqrt(2 / self.nsamples) * inp.float()
        
        if torch.cuda.is_available:
            inp = inp.to('cuda')
        
        out = inp.matmul(inp.t()).to('cpu')
        self.H += out
        inp = inp.to('cpu')
        torch.cuda.empty_cache()
    
    def invert_H(self):
        H = self.H
        self.H = None
        dead = torch.diag(H) == 0
        H[dead, dead] = 1

        damp = self.percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.ncolumns)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        return Hinv, dead

    def compute_stat(self, weight):
        Hinv, dead = self.invert_H()

        W = weight.clone()
        W[:, dead] = 0
        Losses = torch.zeros(torch.tensor(self.ncolumns))

        if torch.cuda.is_available:
            Hinv = Hinv.to('cuda')
            W = W.to('cuda')

        for i in range(0, self.ncolumns):
            w = W[:, i]
            d = Hinv[i, i]
            
            w_dq = self.quantizer.quantize(w, dequantize=True)
            
            Loss = (w - w_dq).dot((w - w_dq)) / d ** 2
            Losses[i] = Loss.to('cpu') / 2

        return Losses
        

def stat_input_hook(m, x, y, estimator):
    if isinstance(x, tuple):
        x = x[0]
    estimator.collect_stat(x)

@torch.no_grad()
def get_weight_scales(model, tokenizer, data_args, bit):
    model.eval()
    device = model.device
    dataset_path = data_args.dataset_path
    num_samples = data_args.num_samples
    seq_len = data_args.max_seq_length

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.shuffle(seed=42)

    hooks = []
    estimators = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            estimator = OBS_estimator(name=name, ncolumns=m.weight.shape[1], device=device)
            hooks.append(
                m.register_forward_hook(functools.partial(stat_input_hook, estimator=estimator))
            )
            estimators[name] = estimator

    for i in tqdm(range(num_samples)):
        print(i)
        input_ids = tokenizer(
            dataset[i]["text"], return_tensors="pt", max_length=seq_len, truncation=True
        ).input_ids.to(device)
        model(input_ids)

    for h in hooks:
        h.remove()

    weight_stats = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            estimator = estimators[name]
            w = m.weight
            estimator.add_sym_quantizer(w, bit)
            losses = estimator.compute_stat(w)
            weight_stats[name] = losses

            del estimator

    return weight_stats


def run_outliers_search(config):

    config_dict = dict(config)
    config_dict['data'] = dict(config_dict['data'])
    config = config_dict

    data_args = DataArguments(
        dataset_path=config['data']['dataset_path'],
        output_path=config['data']['output_path'],
        max_seq_length = config['data']['max_seq_length'],
        num_samples=config['data']['num_samples'],
        trust_remote_code = config['data']['trust_remote_code'],
        preprocessing_num_workers = config['data']['preprocessing_num_workers']
    )

    model_args = ModelArguments(
        model_name_or_path = config['model_name_or_path'], #"/home/projects/LLaMA/huggingface/Llama-2-7b-hf",
        use_fast_tokenizer = config['use_fast_tokenizer'],
        token = config['token'], #None
    )

    model, tokenizer = get_model_and_tokenizer(model_args)

    weight_stats = get_weight_scales(
        model, tokenizer, 
        data_args=data_args, 
        bit=config['bit']
    )

    output_path = Path(data_args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weight_stats, output_path)


def main():
    parser = ArgumentParser()
    parser.add_argument("--config_path", help="path_to_conifg", required=True)

    args = parser.parse_args()
    config = read_config(args.config_path, 'model_configs')

    run_outliers_search(
        config
    )

if __name__ == '__main__':
    main()