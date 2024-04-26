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

    # model = transformers.LlamaForCausalLM.from_pretrained(model_args.model_name_or_path,
    #                                                       torch_dtype=torch.bfloat16,
    #                                                       low_cpu_mem_usage=True)

    return model, tokenizer

def get_dataloader(data_args, tokenizer):
    dataset_path = data_args.dataset_path
    num_samples = data_args.num_samples
    seq_len = data_args.max_seq_length

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.shuffle(seed=42)

    dataloader = []
    for i in tqdm(range(num_samples)):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        input_ids = tokenizer(
            dataset[i]["text"], return_tensors="pt", 
            max_length=seq_len, truncation=True,
            padding="max_length" 
        ).input_ids
        target_ids = input_ids.clone()
        target_ids[:, :-1] = -100
        dataloader.append((input_ids, target_ids))
    return dataloader

def get_wikitext2(data_args, tokenizer):
    nsamples = data_args.num_samples
    seqlen = data_args.max_seq_length
    
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    # testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    # testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(42)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader
    
    

def find_layers(module, layers=[torch.nn.Linear,
                                transformers.models.falcon.modeling_falcon.FalconLinear], name=''):
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

        if self.alpha_scale.device != device:
            self.alpha_scale = self.alpha_scale.to(device)
            self.bit = self.bit.to(device)

        alpha = self.alpha_scale
        bit = self.bit

        # if alpha.device != device:
        #     alpha = alpha.to(device)
        # if bit.device != device:
        #     bit = bit.to(device)

        out_features = w.shape[0]
        
        alpha, qmax, qmin = self._get_row_scale(w, bit)
        alpha = alpha.to(w.dtype)
        self.alpha_scale.data = alpha.reshape((out_features, 1))
        self.qmax = qmax
        self.qmin = qmin

    def _get_row_scale(self, w, bit, maxshrink=0.8, grid=100, norm=2):
        qmax = 1
        qmin = -1
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
        alpha = self.alpha_scale
        qmax = self.qmax
        qmin = self.qmin

        # alpha = alpha.flatten()
        delta = alpha / qmax
        q = torch.round(w / delta)
        q = torch.clamp(q, qmin, qmax)

        if dequantize:
            q = delta * q 

        return q

# class OBS_Estimator:
#     def __init__(
#         self,
#         name,
#         ncolumns,
#         device
#     ):
#         self.name = name
#         self.ncolumns = ncolumns
#         self.device = device
#         self.percdamp = .01 #.025 for max activations
#         self.H = torch.zeros((self.ncolumns, self.ncolumns), device=self.device)
#         self.scaler_row = torch.zeros(self.ncolumns, device=self.device)
#         self.nsamples = 0
#         self.quantizer = None

#     def add_sym_quantizer(self, weight, bit):
#         self.quantizer = SymQuant(out_features=self.ncolumns, bit=bit)
#         self.quantizer.compute_alpha_scale(weight)

#     def add_batch(self, inp, *args):
#         tmp = inp.shape[0]
#         inp = inp.reshape((-1, inp.shape[-1]))    
#         inp = inp.t() #transpose to match computing with analytical formulas

#         self.H *= self.nsamples / (self.nsamples + tmp)
#         self.nsamples += tmp
#         inp = np.sqrt(2 / self.nsamples) * inp.float()
        
#         out = inp.matmul(inp.t())
#         self.H += out

#         # inp = inp.reshape((-1, inp.shape[-1]))
#         # inp = inp.type(torch.float32).abs()
#         # inp_union = torch.vstack([inp, self.scaler_row])
#         # self.scaler_row = torch.max(inp_union, dim=0)[0]
    
#     def invert_H(self):
#         H = self.H.to(self.device)
#         self.H = None
#         dead = torch.diag(H) == 0
#         H[dead, dead] = 1

#         damp = self.percdamp * torch.mean(torch.diag(H))
#         diag = torch.arange(self.ncolumns)
#         H[diag, diag] += damp
#         H = torch.linalg.cholesky(H)
#         H = torch.cholesky_inverse(H)
#         H = torch.linalg.cholesky(H, upper=True)
#         Hinv = H

#         return Hinv, dead

#     def compute_stat(self, weight):
#         # activation_data = self.scaler_row.reshape((1,-1))
#         # self.H = torch.matmul(activation_data.t(), activation_data)

#         Hinv, dead = self.invert_H()

#         W = weight.clone()
#         W[:, dead] = 0
#         Losses = torch.zeros(torch.tensor(self.ncolumns))

#         if W.device != self.device:
#             W = W.to(self.device)
#         # if torch.cuda.is_available:
#         #     Hinv = Hinv.to('cuda')
#         #     W = W.to('cuda')

#         for i in range(0, self.ncolumns):
#             w = W[:, i].unsqueeze(1)
#             d = Hinv[i, i]
            
#             if self.quantizer is not None:
#                 w_dq = self.quantizer.quantize(w, dequantize=True)
#                 Loss = torch.matmul((w - w_dq).t(), (w - w_dq)) / d**2
#             else:
#                 Loss = torch.matmul(w.t(), w) / d**2
            
#             Losses[i] = Loss[0] / 2

#         Losses = Losses.cpu()
#         return Losses


# class OBS_Estimator:
#     def __init__(
#         self,
#         name,
#         ncolumns,
#         device
#     ):
#         self.name = name
#         self.ncolumns = ncolumns
#         self.device = device
#         self.percdamp = .01 #.025 for max activations
#         self.H = torch.zeros((self.ncolumns, self.ncolumns), device=self.device)
#         self.scaler_row = torch.zeros(self.ncolumns, device=self.device)
#         self.nsamples = 0
#         self.quantizer = None

#     def add_sym_quantizer(self, weight, bit):
#         self.quantizer = SymQuant(out_features=self.ncolumns, bit=bit)
#         self.quantizer.compute_alpha_scale(weight)


#     def add_batch(self, inp, *args):
#         tmp = inp.shape[0]
#         inp = inp.reshape((-1, inp.shape[-1]))    
#         inp = inp.t() #transpose to match computing with analytical formulas

#         self.H *= self.nsamples / (self.nsamples + tmp)
#         self.nsamples += tmp
#         inp = np.sqrt(2 / self.nsamples) * inp.float()
        
#         out = inp.matmul(inp.t())
#         self.H += out

#         # inp = inp.reshape((-1, inp.shape[-1]))
#         # inp = inp.type(torch.float32).abs()
#         # inp_union = torch.vstack([inp, self.scaler_row])
#         # self.scaler_row = torch.max(inp_union, dim=0)[0]

#     def compute_stat(self, weight):
#         # activation_data = self.scaler_row.reshape((1,-1))
#         # self.H = torch.matmul(activation_data.t(), activation_data)

#         W = weight.clone()
#         Losses = torch.zeros(torch.tensor(self.ncolumns))

#         if W.device != self.device:
#             W = W.to(self.device)
#         # if torch.cuda.is_available:
#         #     Hinv = Hinv.to('cuda')
#         #     W = W.to('cuda')

#         W_dq = self.quantizer.quantize(W, dequantize=True)
#         frob_norm_error = (W - W_dq).pow(2).sum(dim=0)

#         Losses = torch.diag(self.H)
#         Losses *= frob_norm_error

#         Losses = Losses.cpu()
#         return Losses


class OBD_Estimator:
    def __init__(
        self,
        name,
        ncolumns,
        device,
        agg
    ):
        self.name = name
        self.ncolumns = ncolumns
        self.device = device
        self.percdamp = .01 #.025 for max activations

        self.nsamples = 0
        self.quantizer = None
        self.agg = agg

        if self.agg == 'l2':
            self.H = torch.zeros((self.ncolumns, self.ncolumns), device=self.device)
        
        elif self.agg == 'max':
            self.scaler_row = torch.zeros(self.ncolumns, device=self.device)

    def add_sym_quantizer(self, weight, bit):
        self.quantizer = SymQuant(out_features=self.ncolumns, bit=bit)
        self.quantizer.compute_alpha_scale(weight)


    def add_batch(self, inp, *args):
        if self.agg == 'l2':
            tmp = inp.shape[0]
            inp = inp.reshape((-1, inp.shape[-1]))    
            inp = inp.t() #transpose to match computing with analytical formulas

            self.H *= self.nsamples / (self.nsamples + tmp)
            self.nsamples += tmp
            inp = np.sqrt(2 / self.nsamples) * inp.float()
            
            out = inp.matmul(inp.t())
            self.H += out
        
        elif self.agg == 'max':
            inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.type(torch.float32).abs()
            inp_union = torch.vstack([inp, self.scaler_row])
            self.scaler_row = torch.max(inp_union, dim=0)[0]

    def compute_stat(self, weight):

        W = weight.clone()
        Losses = torch.zeros(torch.tensor(self.ncolumns))

        if W.device != self.device:
            W = W.to(self.device)
        # if torch.cuda.is_available:
        #     Hinv = Hinv.to('cuda')
        #     W = W.to('cuda')

        W_dq = self.quantizer.quantize(W, dequantize=True)

        if self.agg == 'l2':            
            err = (W - W_dq).pow(2).sum(dim=0)
            Losses = torch.diag(self.H)
        
        elif self.agg == 'max':
            H = self.scaler_row * self.scaler_row
            err = torch.abs(W - W_dq)
            err = torch.max(err, dim=0)[0]
            Losses = H

        Losses *= err
        Losses = Losses.cpu()

        return Losses

class OBDx2_Estimator:
    def __init__(
        self,
        name,
        ncolumns,
        device,
        agg
    ):
        self.name = name
        self.ncolumns = ncolumns
        self.device = device
        self.percdamp = .01 #.025 for max activations

        self.nsamples = 0
        self.quantizer = None
        self.agg = agg

        if self.agg == 'l2':
            self.H = torch.zeros((self.ncolumns, self.ncolumns), device=self.device)
        
        elif self.agg == 'max':
            self.scaler_row = torch.zeros(self.ncolumns, device=self.device)

    def add_sym_quantizer(self, weight, bit):
        self.quantizer = SymQuant(out_features=self.ncolumns, bit=bit)
        self.quantizer.compute_alpha_scale(weight)


    def add_batch(self, inp, *args):
        if self.agg == 'l2':
            tmp = inp.shape[0]
            inp = inp.reshape((-1, inp.shape[-1]))    
            inp = inp.t() #transpose to match computing with analytical formulas

            self.H *= self.nsamples / (self.nsamples + tmp)
            self.nsamples += tmp
            inp = np.sqrt(2 / self.nsamples) * inp.float()
            
            out = inp.matmul(inp.t())
            self.H += out
        
        elif self.agg == 'max':
            inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.type(torch.float32).abs()
            inp_union = torch.vstack([inp, self.scaler_row])
            self.scaler_row = torch.max(inp_union, dim=0)[0]

    def compute_stat(self, weight):

        W = weight.clone()
        Losses = torch.zeros(torch.tensor(self.ncolumns))

        if W.device != self.device:
            W = W.to(self.device)
        # if torch.cuda.is_available:
        #     Hinv = Hinv.to('cuda')
        #     W = W.to('cuda')

        W_dq = self.quantizer.quantize(W, dequantize=True)

        if self.agg == 'l2':            
            err = (W - W_dq).pow(2).sum(dim=0)
            Losses = torch.diag(self.H)
        
        elif self.agg == 'max':
            H = self.scaler_row * self.scaler_row
            err = torch.abs(W - W_dq)
            err = torch.max(err, dim=0)[0]
            err = err ** 2
            Losses = H

        Losses *= err
        Losses = Losses.cpu()

        return Losses


class Wanda_Estimator:
    def __init__(
        self,
        name,
        ncolumns,
        device,
        agg
    ):   
        self.name = name
        self.ncolumns = ncolumns
        self.device = device
        self.percdamp = .01
        self.agg = agg #'l2', 'max' 
        self.scaler_row = torch.zeros(self.ncolumns, device=self.device)
        self.nsamples = 0
        self.quantizer = None
    
    def add_sym_quantizer(self, weight, bit):
        self.quantizer = SymQuant(out_features=self.ncolumns, bit=bit)
        self.quantizer.compute_alpha_scale(weight)

    def add_batch(self, inp, *args):
        # tmp = inp.shape[0]
        # inp = inp.reshape((-1, inp.shape[-1]))
        # inp = inp.t()

        # self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        # self.nsamples += tmp

        # inp = inp.type(torch.float32)
        # self.scaler_row += torch.norm(inp, p=2, dim=1)**2  / self.nsamples

        inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.type(torch.float32).abs()
        inp_union = torch.vstack([inp, self.scaler_row])
        self.scaler_row = torch.max(inp_union, dim=0)[0]
             
    def compute_stat(self, w):
        
        if self.quantizer is not None:
            w_dq = self.quantizer.quantize(w, dequantize=True)
            Loss = torch.abs(w - w_dq)
        else:
            Loss = torch.abs(w)

        if self.agg == 'max':
            activation_data = self.scaler_row.reshape((1,-1))
        elif self.agg == 'max_sqrt':
            activation_data = torch.sqrt(self.scaler_row.reshape((1,-1)))
        
        Loss *= activation_data
        Loss = torch.max(Loss, dim=0)[0]        

        Loss = Loss.cpu()
        return Loss


        # activation_data = torch.sqrt(self.scaler_row.reshape((1,-1)))
        # activation_data = self.scaler_row.reshape((1,-1))

        # if self.quantizer is not None:
        #     w_dq = self.quantizer.quantize(w, dequantize=True)
        #     Loss = torch.abs(w - w_dq) * activation_data
        # else:
        #     Loss = torch.abs(w) * activation_data

        # if self.agg == 'l2':
        #     Loss = torch.norm(Loss, p=2, dim=0)
        # elif self.agg == 'max':
        #     Loss = torch.max(Loss, dim=0)[0]

        # Loss = Loss.cpu()
        
        # return Loss
        
class Activation_Estimator:
    def __init__(
        self,
        name,
        ncolumns,
        device
    ):
        self.name = name
        self.ncolumns = ncolumns
        self.device = device
        self.scaler_row = torch.zeros(self.ncolumns, device=self.device)
        self.nsamples = 0
        self.quantizer = None

    def add_batch(self, inp, *args):
        inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.type(torch.float32).abs()
        inp_union = torch.vstack([inp, self.scaler_row])
        self.scaler_row = torch.max(inp_union, dim=0)[0]
    
    def compute_stat(self, *args):
        return self.scaler_row.cpu()
    

# class Output_Estimator:
#     def __init__(
#         self,
#         name,
#         nrows,
#         ncolumns,
#         device,
#         agg
#     ):
#         self.name = name
#         self.nrows = nrows
#         self.ncolumns = ncolumns
#         self.device = device
#         self.percdamp = .01
#         self.agg = agg #'l2', 'max' 
#         self.Losses = torch.zeros(self.nrows, device=self.device)
#         self.nsamples = 0
#         self.quantizer = None

#     def add_sym_quantizer(self, weight, bit):
#         self.quantizer = SymQuant(out_features=self.ncolumns, bit=bit)
#         self.quantizer.compute_alpha_scale(weight)
    
#     def compute_loss(self, tmp, out, out_dq=None):
#         if self.agg == 'l2':
#             self.Losses *= self.nsamples / (self.nsamples+tmp)
#             self.nsamples += tmp

#             if out_dq is None:
#                 self.Losses += torch.norm(out, p=2, dim=0) / self.nsamples
#             else:
#                 self.Losses += torch.norm((out - out_dq), p=2, dim=0) / self.nsamples
        
#         elif self.agg == 'max':
#             if out_dq is None:
#                 loss = torch.vstack([torch.abs(out), self.Losses])
#                 self.Losses = torch.max(loss, dim=0)[0]
#             else:
#                 loss = torch.vstack([torch.abs(out - out_dq), self.Losses])
#                 self.Losses = torch.max(loss, dim=0)[0]

#     def add_batch(self, inp, out, l): 
#         tmp = inp.shape[0]
#         w = l.weight

#         inp = inp.reshape((-1, inp.shape[-1]))
#         # out = out.reshape((-1, out.shape[-1]))
#         inp_nrows = inp.shape[0]
#         out_w = torch.zeros((inp_nrows, self.ncolumns), device=self.device)
#         for i in range(inp_nrows):
#             out_w = inp[i] * w

#         if self.quantizer is not None:
            
#             w_dq = self.quantizer.quantize(w, dequantize=True)
#             out_dq = F.linear(inp, w_dq)
#             # Loss = (out - out_dq) * (out - out_dq)
#             # Loss = torch.mean(Loss, dim=0)
#             self.compute_loss(tmp, out, out_dq)
#         else:
#             self.compute_loss(tmp, out)

#     def compute_stat(self, *args):
#         return self.Losses.cpu()


@torch.no_grad()
def llama_sequential(model, dataloader, data_args, estimator_args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    nsamples = data_args.num_samples
    bit = estimator_args['bit']
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(device)
    model.model.norm = model.model.norm.to(device)
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (data_args.num_samples, data_args.max_seq_length, model.config.hidden_size), dtype=dtype, device=device
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    weight_stats = {}
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

            estimators = {}
            for name in subset:
                print(f'{name}', end='  ', flush=True)
                
                lin_layer = subset[name]
                nrows = lin_layer.weight.shape[0]
                ncolumns = lin_layer.weight.shape[1]
                if estimator_args['estimator'] == 'OBD_Estimator':
                    agg = estimator_args['agg']
                    estimator = OBD_Estimator(
                        name=name, ncolumns=ncolumns, device=device, agg=agg
                    )
                elif estimator_args['estimator'] == 'OBDx2_Estimator':
                    agg = estimator_args['agg']
                    estimator = OBDx2_Estimator(
                        name=name, ncolumns=ncolumns, device=device, agg=agg
                    )
                elif estimator_args['estimator'] == 'Wanda_Estimator':
                    agg = estimator_args['agg']
                    estimator = Wanda_Estimator(
                        name=name, ncolumns=ncolumns, device=device, agg=agg
                    ) 
                elif estimator_args['estimator'] == 'Activation_Estimator':
                    estimator = Activation_Estimator(
                        name=name, ncolumns=ncolumns, device=device
                    )  
                # elif estimator_args['estimator'] == 'Output_Estimator':
                #     agg = estimator_args['agg']
                #     estimator = Output_Estimator(
                #         name=name, nrows=nrows, ncolumns=ncolumns, device=device, agg=agg
                #     )

                if estimator_args['add_quantizer']:
                    w = lin_layer.weight
                    estimator.add_sym_quantizer(w, bit)

                estimators[name] = estimator

            def add_batch(name):
                def tmp(l, inp, out):
                    estimators[name].add_batch(inp[0].data, out.data, l)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(nsamples):
                outs[j] = decoder_layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                lin_layer = subset[name]
                w = lin_layer.weight

                estimator = estimators[name]

                losses = estimator.compute_stat(w)
                weight_stats[f'model.layers.{i}.{name}'] = losses
                # del estimator

        for j in range(nsamples):
            outs[j] = decoder_layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = decoder_layer.cpu()
        del decoder_layer
        del estimators
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    
    return weight_stats


def run_outliers_search(config):

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

    dataloader = get_dataloader(data_args, tokenizer)
    # dataloader = get_wikitext2(data_args, tokenizer)
    

    weight_stats = llama_sequential(model, dataloader, data_args, estimator_args)


    # weight_stats = get_weight_scales(
    #     model, tokenizer, 
    #     data_args=data_args, 
    #     bit=config['bit']
    # )

    output_path = Path(data_args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weight_stats, output_path)

    print('', flush=True)
    print(f'weight_stats have been saved in {output_path}', flush=True)

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
