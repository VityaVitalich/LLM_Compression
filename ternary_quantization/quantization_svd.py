import os
import math
import random
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

class SymQuant:
    def __init__(
        self,
        bit
    ):

        self.bit = torch.tensor(bit)
        
        self.alpha_scale = None
        self.qmin = None
        self.qmax = None

    def compute_qlimits(self):
        if self.bit == 1.58:
            self.qmax = torch.tensor(1)
            self.qmin = torch.tensor(-1)
        elif self.bit == 2:
            self.qmax = torch.tensor(1)
            self.qmin = torch.tensor(-2)
        elif self.bit == 3:
            self.qmax = torch.tensor(3)
            self.qmin = torch.tensor(-4)
        elif self.bit == 4:
            self.qmax = torch.tensor(7)
            self.qmin = torch.tensor(-8)
        elif self.bit == 8:
            self.qmax = torch.tensor(127)
            self.qmin = torch.tensor(-128)
        else:
            raise ValueError('Only ternary and 2/4/8-bit is supported!')


    def compute_alpha_scale(self, quant_weight) -> None:
        w = quant_weight.data
        device = quant_weight.device

        if self.bit.device != device:
            self.bit = self.bit.to(device)
            self.qmin = self.qmin.to(device)
            self.qmax = self.qmax.to(device)

        bit = self.bit
        out_features = w.shape[0]
        
        alpha = self._get_row_scale(w, bit)
        alpha = alpha.to(w.dtype)

        self.alpha_scale = alpha.reshape((out_features, 1))


    def _get_row_scale(self, w, bit, maxshrink=0.8, grid=100, norm=2):
        qmax = self.qmax
        qmin = self.qmin

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

        return alpha
    

    def quantize(self, weight, dequantize=True, calib_scale=True):
        # if torch.cuda.is_available:
        #     w = weight.to('cuda')
        #     alpha = self.alpha_scale.to('cuda')
        #     qmax = self.qmax.to('cuda')
        #     qmin = self.qmin.to('cuda')
        # else:
        #     w = weight
        #     alpha = self.alpha_scale
        #     qmax = self.qmax
        #     qmin = self.qmin

        w = weight

        if calib_scale:
            alpha = self.alpha_scale
        else:
            alpha = w.abs().max(dim=0)[0].reshape(-1, 1)
        
        qmax = self.qmax
        qmin = self.qmin

        delta = alpha / qmax
        q = torch.round(w / delta)
        q = torch.clamp(q, qmin, qmax)

        if dequantize:
            q = delta * q 

        return q

class SVD_Estimator:
    def __init__(
        self,
        name,
        ncolumns,
        device,
        rank_reduction = 1,
        fp_indices = 0
    ):
        self.name = name
        self.ncolumns = ncolumns
        self.device = device
        self.rank_reduction = rank_reduction
        self.fp_indices = fp_indices

        self.quantizer = None
        self.mask = None
        

    def add_sym_quantizer(self, weight, bit):
        self.quantizer = SymQuant(bit=bit)
        self.quantizer.compute_qlimits()

    def _get_mask(self, w):
        self.mask = torch.ones(self.ncolumns, 
                               dtype=torch.bool,
                               device=self.device)
        self.mask[self.fp_indices] = False

        # w_dq = self.quantizer.quantize(w, dequantize=True, calib_scale=False)
        # mask = (w_dq != 0.0)

    #None, gesvd, gesvdj, and gesvda
    def _weight_svd(self, w, new_rank=64, driver=None):
        w_dtype = w.dtype
        U, S, Vh = torch.linalg.svd(w.float(), full_matrices=True, driver=driver)
        U = U[:, :new_rank]
        S = S[:new_rank]
        U = U @ torch.diag(S)
        Vh = Vh[:new_rank, :]
        U = U.to(w_dtype)
        Vh = Vh.to(w_dtype)
        return U, Vh

    def estimate(self, weight):
        w = weight
        out_features = w.shape[0]
        self._get_mask(w)
        
        w_quant = self.mask * w

        #gesvd, gesvdj, and gesvda
        new_rank = int(out_features / self.rank_reduction)
        U, Vh = self._weight_svd(w_quant, new_rank=new_rank)

        self.quantizer.compute_alpha_scale(U)
        U_dq = self.quantizer.quantize(U, dequantize=True, calib_scale=True)

        self.quantizer.compute_alpha_scale(Vh)
        Vh_dq = self.quantizer.quantize(Vh, dequantize=True, calib_scale=True)        

        w_dq = U_dq @ Vh_dq

        w_dq[:, ~self.mask] = w[:, ~self.mask]
        err = torch.abs(w - w_dq)
        return w_dq, err


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

    # model = transformers.LlamaForCausalLM.from_pretrained(model_args.model_name_or_path,
    #                                                       torch_dtype=torch.bfloat16,
    #                                                       low_cpu_mem_usage=True)

    return model, tokenizer

def find_layers(module, layers=[torch.nn.Linear], name=''):
    for layer in layers:
        if isinstance(module, layer):
            return {name: module}
    # if type(module) in layers:
    #     return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

@torch.no_grad()
def extract_quant_err(model, estimator_args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # nsamples = data_args.num_samples
    bit = estimator_args['bit']
    rank_reduction = estimator_args['rank_reduction']
    fp_features = estimator_args['fp_features']
    path_to_cols_metric = estimator_args['path_to_cols_metric']
    if path_to_cols_metric is not None:
        layers_scale = torch.load(path_to_cols_metric)

    layers = model.model.layers

    for i in range(len(layers)):
        print(f'\nLayer {i}:', flush=True, end=' ')
        decoder_layer = layers[i].to(device)
        full = find_layers(decoder_layer)

        sequential = [
            ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
            ['self_attn.o_proj'],
            ['mlp.up_proj', 'mlp.gate_proj'],
            ['mlp.down_proj']
        ]

        for names in sequential:
            subset = {n: full[n] for n in names}

            for name in subset:
                print(f'{name}', end='  ', flush=True)

            lin_layer = subset[name]
            nrows = lin_layer.weight.shape[0]
            ncolumns = lin_layer.weight.shape[1]
            w = lin_layer.weight

            if fp_features > 0:
                layer_scale = layers_scale['model.layers.{}.{}'.format(i, name)]
                fp_indices = torch.sort(layer_scale)[1][-fp_features:]

            estimator = SVD_Estimator(
                name=name, ncolumns=ncolumns, device=device, 
                fp_indices=fp_indices, rank_reduction=rank_reduction)
            estimator.add_sym_quantizer(w, bit)
            w_dq, err = estimator.estimate(w)
            w.data = w_dq

def run_quant(config):

    config_dict = dict(config)
    config_dict['data'] = dict(config_dict['data'])
    config_dict['estimator'] = dict(config_dict['estimator'])
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

    estimator_args = config['estimator']

    model, tokenizer = get_model_and_tokenizer(model_args)

    extract_quant_err(model, estimator_args)



    # dataloader = get_dataloader(data_args, tokenizer)
    # dataloader = get_wikitext2(data_args, tokenizer)
    

    # weight_stats = llama_sequential(model, dataloader, data_args, estimator_args)


    # weight_stats = get_weight_scales(
    #     model, tokenizer, 
    #     data_args=data_args, 
    #     bit=config['bit']
    # )

    output_path = Path(data_args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    # torch.save(weight_stats, output_path)

    print('', flush=True)
    print(f'model have been saved in {output_path}', flush=True)    

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
    run_quant(config)

if __name__ == '__main__':
    main()