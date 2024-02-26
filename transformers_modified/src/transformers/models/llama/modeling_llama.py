# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import math
import warnings
from typing import List, Optional, Tuple, Union, Literal
from operator import attrgetter

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from ...utils.import_utils import is_torch_fx_available
from .configuration_llama import LlamaConfig


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    warnings.warn(
        "Calling `transformers.models.llama.modeling_llama._prepare_4d_attention_mask` is deprecated and will be removed in v4.37. Use `transformers.modeling_attn_mask_utils._prepare_4d_attention_mask"
    )
    return _prepare_4d_attention_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    warnings.warn(
        "Calling `transformers.models.llama.modeling_llama._make_causal_mask` is deprecated and will be removed in v4.37. Use `transformers.models.llama.modeling_llama.AttentionMaskConverter._make_causal_mask"
    )
    return AttentionMaskConverter._make_causal_mask(
        input_ids_shape=input_ids_shape, dtype=dtype, device=device, past_key_values_length=past_key_values_length
    )

def clipped_softmax(data, dim=-1, eta=1.1, gamma=-0.1, **kw):
    sm_out = torch.nn.functional.softmax(data, dim=dim, **kw)
    if (eta == 1) and (gamma == 0):
        return sm_out
    stretched_out = sm_out * (eta - gamma) + gamma
    return torch.clip(stretched_out, 0, 1)

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, eta=1, gamma=0) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = clipped_softmax(attn_weight, dim=-1, eta=eta, gamma=gamma)
    #attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def quantize(
    X: torch.Tensor = None,
    B: int = 16,
    ) -> torch.Tensor:

    thd_neg = -(2 ** (B - 1)) + 1
    thd_pos = 2 ** (B - 1) - 1


    scale = (X.max() - X.min())/(thd_neg - thd_pos)
    X = round_pass(X/scale)
    X = torch.clip(X, thd_neg, thd_pos)
    return scale*X

@torch.jit.script
def quantize_over_blocks(
    X: torch.Tensor,
    B: int = 16,
    block_size: int = 4,  # Assuming block size along the first dimension
) -> torch.Tensor:
    # Dimensions for the input tensor
    D = X.shape[1]

    # Quantization thresholds
    thd_neg = -(2 ** (B - 1)) + 1
    thd_pos = 2 ** (B - 1) - 1
    # Initialize an output tensor
    X_quantized = torch.zeros_like(X)
    
    # Calculate number of blocks
    num_blocks = (D + block_size - 1) // block_size  # Account for the last block that might be smaller
    
    for i in range(num_blocks):
        # Extract the block
        start_idx = i * block_size
        end_idx = min((i + 1) * block_size, D)
        block = X[:,start_idx:end_idx]
        
        # Scale for the current block
        scale = (block.max() - block.min()) / (thd_pos - thd_neg)
        block = round_pass(block / scale)
        block = torch.clip(block, thd_neg, thd_pos)
        
        # Store the quantized block back into the tensor
        X_quantized[:,start_idx:end_idx] = scale * block
    
    return X_quantized

def quantize_with_outliers(X: torch.Tensor,
    B: int = 16,
    block_size: int = 4,
    idx: torch.Tensor = torch.tensor([])):

    if len(idx) == 0:
        print('Empty outlier idx')
        return quantize_over_blocks(X, B=B, block_size=block_size)
    
    mask = torch.ones(X.size(1), dtype=torch.bool)
    mask[idx] = False

    # Split the tensor into quantize and no_quantize parts
    X_quantize = X[:, mask]
    X_no_quantize = X[:, ~mask]

    # Quantize the part that needs quantization
    X_quantized = quantize_over_blocks(X_quantize, B=B, block_size=block_size)

    # # Prepare a tensor to hold the result
    # X_result = torch.empty_like(X)

    # Place the quantized and unquantized parts back in their original positions
    X[:, mask] = X_quantized
    X[:, ~mask] = X_no_quantize

    return X


class LsqQuant(nn.Module):
    def __init__(
        self, bit, all_positive=False, symmetric=False, per_channel=True, outlier_ids=[]
    ):
        super(LsqQuant, self).__init__()
        self.bit = bit
        self.outlier_ids = outlier_ids


        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2**bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = -(2 ** (bit - 1)) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = -(2 ** (bit - 1))
                self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel
        self.s = nn.Parameter(torch.ones(1))

    def init_from(self, x, *args, **kwargs):
        self.mask = torch.ones(x.size(1), dtype=torch.bool)
        self.mask[self.outlier_ids] = False
        x_quantize = x[:, self.mask]

        self.s = nn.Parameter(
                x_quantize.detach().abs().mean(dim=0) * 2 / (self.thd_pos**0.5)
        )

    def forward(self, x):
        if self.bit >= 32:
            return x


        x_result = torch.empty_like(x)        
        x_quantize = x[:, self.mask]

        s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        
        device = x_quantize.device
        s_scale = grad_scale(self.s, s_grad_scale).to(device)
        x_quantize = x_quantize / (s_scale)
        x_quantize = torch.clamp(x_quantize, self.thd_neg, self.thd_pos)
        x_quantize = round_pass(x_quantize)
        x_quantize = x_quantize * (s_scale)

        #assert (x_quantize == x[:, self.mask]).all()
        x_result[:, self.mask] = x_quantize
        x_result[:, ~self.mask] = x[:, ~self.mask]
        return x_result


class NoiseQuant(nn.Module):
    def __init__(
        self,
        weight_shape,
        bit, 
        block_size,
        quant_cols_num,
        mask
    )-> None:
        super(NoiseQuant, self).__init__()
        self.weight_shape = weight_shape

        self.bit = bit
        self.block_size = block_size
        self.mask = mask

        # self.quant_scale = None

        # self.register_buffer('quant_scale', torch.ones(weight_shape[1]).unsqueeze(1))
        quant_scale = torch.ones((weight_shape[0], quant_cols_num))
        self.register_buffer('quant_scale', quant_scale)

    def compute_quant_scale(self, weight) -> None:
        assert self.weight_shape == weight.shape, 'Shape of input weight is incompatible!'

        w = weight.data.clone()
        bit = self.bit
        block_size = self.block_size
        
        if self.mask is not None:
            w = w[:, self.mask]
        
        quanted_features = w.shape[1]

        if (block_size == 0):
            scale = self._get_row_scale(w, bit)
            scale = scale.unsqueeze(1)
        else:
            scale = []
            for i in range(0, quanted_features, block_size):
                w_block = w[:, i:(i + block_size)]
                scale_block = self._get_row_scale(w_block, bit)
                scale.append(scale_block)

            scale = torch.vstack(scale).T

        scale = scale.to(w.dtype)
        self.quant_scale = scale.contiguous()
        

    def _get_row_scale(self, w, bit, maxshrink=0.8, grid=100, norm=2):
        qmax = 2 ** (bit - 1) - 1
        qmin = -(2 ** (bit - 1))
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

        scale = wmax / qmax

        for i in range(int(maxshrink * grid)):
            p = 1 - i / grid 
            wmax1 = p * wmax

            scale1 = wmax1 / qmax

            #quantization
            q = torch.clamp(torch.round(w / scale1.unsqueeze(1)), qmin, qmax)
            #dequantization
            q = q * scale1.unsqueeze(1)

            q -= w
            q.abs_()
            q.pow_(norm)
            err = torch.sum(q, 1)
            tmp = err < best

            if torch.any(tmp):
                best[tmp] = err[tmp]
                scale[tmp] = scale1[tmp]

        return scale

    def quant_noise(self, weight):
        assert self.weight_shape == weight.shape, 'Shape of input weight is incompatible!'

        w = weight.data
        device = weight.device
        block_size = self.block_size
        in_features = self.weight_shape[1]
        scale = self.quant_scale
        mask = self.mask

        if scale.device != device:
            scale = scale.to(device)

        if mask is not None:
            w_rand = torch.randn_like(w[:, mask], requires_grad=False) / 2
        else:
            w_rand = torch.randn_like(w, requires_grad=False) / 2

        if block_size == 0:
            # scale = torch.repeat_interleave(scale, in_features, dim=1)
            w_rand = scale * w_rand
        elif block_size > 0:
            scale = torch.repeat_interleave(scale, block_size, dim=1)
            w_rand = scale * w_rand

        if mask is not None:
            w_rand_tmp = torch.zeros(w.shape, dtype=w.dtype, device=w.device)
            w_rand_tmp[:, mask] = w_rand
            w_rand = w_rand_tmp

        return w_rand

    def forward(self, weight):
        w_rand = self.quant_noise(weight)
        w = weight + w_rand
        
        return w


class BitNoiseQuant(nn.Module):
    def __init__(
        self,
        weight_shape,
        bit,
        block_size,
        quant_cols_num,
        mask
    )-> None:

        super(BitNoiseQuant, self).__init__()
        self.weight_shape = weight_shape

        self.bit = torch.tensor(bit)
        self.block_size = block_size
        self.quant_cols_num = quant_cols_num
        self.mask = mask

        # alpha = torch.ones((weight_shape[0], self.quant_cols_num))

        alpha = torch.ones((self.weight_shape[0] * self.quant_cols_num, 1))
        self.alpha = nn.Parameter(alpha)
        # self.register_buffer('alpha', alpha)



    def compute_alpha_scale(self, weight) -> None:
        assert self.weight_shape == weight.shape, 'Shape of input weight is incompatible!'

        w = weight.data
        device = weight.device
        block_size = self.block_size
        quant_cols_num = self.quant_cols_num
        alpha = self.alpha
        bit = self.bit
        mask = self.mask

        if alpha.device != device:
            alpha = alpha.to(device)
        if bit.device != device:
            bit = bit.to(device)
        if mask.device != device:
            mask = mask.to(device)

        if mask is not None:
            w_re = w[:, mask]
        else:
            w_re = w

        out_features = w_re.shape[0]
        in_features = w_re.shape[1]

        if block_size > 0:
            # w_re = w_re.reshape((out_features * block_size, in_features // block_size))
            w_re = w_re.reshape((out_features * quant_cols_num, block_size))

        
        alpha = self._get_row_scale(w_re, bit)
        alpha = alpha.to(w.dtype)
        self.alpha.data = alpha.reshape((out_features * quant_cols_num, 1))

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

        return alpha

    def _lsq_forward(self, w, bit, alpha):
        qmax = 2 ** (bit.detach() - 1) - 1
        # q = F.hardtanh(w / alpha, -1.0, 1.0) * qmax
        q = F.hardtanh(w / alpha, -1.0, 1.0)
        mask_q_pos = (q > 0)
        q = q * (2 ** (bit.detach() - 1)) - mask_q_pos * q
        out = (q.round() + (q - q.detach())) * (alpha / qmax)
        return out

    def _compute_quant_noise(self, w, bit, alpha):
        # bit = nn.Parameter(torch.zeros(1))
        # alpha = nn.Parameter(torch.tensor(0.01))

        N_BIN = 256
        # bit = 2 + torch.sigmoid(bit)*4
        # bit = 1.5 + torch.sigmoid(bit)

        # bit += (torch.rand_like(bit) - 0.5)
        # bit = bit.round() + (bit - bit.detach())

        alpha = F.softplus(alpha, beta=10**(6), threshold=1) 
        lsq = self._lsq_forward(w, bit.round(), alpha)

        c1 = w >= alpha
        c2 = w <= -alpha     
        delta = alpha / (2**(bit - 1) - 1)

        with torch.no_grad():                
            diff = (lsq - w) / delta #difference between dequantized and original weights after their scale
            sel = diff[torch.logical_not(torch.logical_or(c1, c2))] #take weights less than alpha
            hist = torch.histc(sel.float(), bins=N_BIN, min=-0.5, max=0.5)    
            
            noise = torch.multinomial(hist, w.numel(), True) + torch.rand_like(w.view(-1))               
            noise = (noise / N_BIN - 0.5).view(w.shape)
            noise = noise.to(w.dtype)

        w_rand = noise * delta
        w_cliped = torch.where(c2, -alpha, w + w_rand)
        w_cliped = torch.where(c1, alpha, w_cliped)
        
        return w_cliped


    def quant_noise(self, weight):
        assert self.weight_shape == weight.shape, 'Shape of input weight is incompatible!'

        w = weight
        device = weight.device
        block_size = self.block_size
        quant_cols_num = self.quant_cols_num
        alpha = self.alpha
        bit = self.bit
        mask = self.mask

        if alpha.device != device:
            alpha = alpha.to(device)
        if bit.device != device:
            bit = bit.to(device)
        if mask.device != device:
            mask = mask.to(device)

        if mask is not None:
            w_re = w[:, mask]
        else:
            w_re = w

        out_features = w_re.shape[0]
        in_features = w_re.shape[1]

        if block_size > 0:
            out_features = w_re.shape[0]
            in_features = w_re.shape[1]
            # w_re = w_re.reshape((out_features * block_size, in_features // block_size))
            w_re = w_re.reshape((out_features * quant_cols_num, block_size))

        w_cliped = self._compute_quant_noise(w_re, bit, alpha)

        if mask is not None:
            w_out = torch.zeros(w.shape, dtype=w.dtype, device=w.device)
            w_out[:, mask] = w_cliped.reshape((out_features, in_features))
            w_out[:, ~mask] = w[:, ~mask]
        else:
            w_out = w_cliped

        return w_out

    def quantize_weight(self, weight):
        assert self.weight_shape == weight.shape, 'Shape of input weight is incompatible!'

        w = weight
        device = weight.device
        block_size = self.block_size
        quant_cols_num = self.quant_cols_num
        alpha = self.alpha
        bit = self.bit
        mask = self.mask

        if alpha.device != device:
            alpha = alpha.to(device)
        if bit.device != device:
            bit = bit.to(device)
        if mask.device != device:
            mask = mask.to(device)

        if mask is not None:
            w_re = w[:, mask]
        else:
            w_re = w

        out_features = w_re.shape[0]
        in_features = w_re.shape[1]
        if block_size > 0:
            # w_re = w_re.reshape((out_features * block_size, in_features // block_size))
            w_re = w_re.reshape((out_features * quant_cols_num, block_size))

        lsq = self._lsq_forward(w_re, bit.round(), alpha)

        if mask is not None:
            q_out = torch.zeros(w.shape, dtype=w.dtype, device=w.device)
            q_out[:, mask] = lsq.reshape((out_features, in_features))
            q_out[:, ~mask] = w[:, ~mask]
        else:
            q_out = lsq

        return q_out




    def forward(self, weight):
        w = self.quant_noise(weight)

        return w

class QuantizedLinear(nn.Linear):

    def __init__(self, 
        linear: nn.Linear, 
        # bit: int = 4,
        # outlier_ids: list = [0],
        # learnable_scales: bool = False,
        training_mode: Literal[
            'train_full', 
            'train_outlier', 
            'train_quant'
        ] = 'train_full'
    ):
        super().__init__(
            in_features=linear.in_features,
            out_features=linear.out_features,
            # bias=(linear.bias is not None),
            device=linear.weight.device, 
            dtype=linear.weight.dtype
        )
        # self.load_state_dict(linear.state_dict())
        self.weight = nn.Parameter(linear.weight.data)
        self.bias = nn.Parameter(linear.bias.data) if linear.bias is not None else None

        self.mask = None
        # self.register_buffer('mask', mask)

        self.quantizer = None
        # self.learnable_scales = learnable_scales
        # if self.learnable_scales:
        #     self.quantizer = LsqQuant(bit=bit, symmetric=True, outlier_ids=outlier_ids)
        #     self.quantizer.init_from(self.weight)

        self.add_quant_noise = None
        self.add_quant_noise_predict = None

        self.add_quant_bitnoise = None
        self.add_quant_bitnoise_predict = None

        self.training_mode = training_mode

        # self.register_buffer('quant_scale', None)

    def set_mask(self, outlier_ids) -> None:
        with torch.no_grad():
            self.mask = torch.ones(self.weight.size(1), dtype=torch.bool)
            self.mask[outlier_ids] = False

    def add_quant_noise_to_weight(
        self, 
        bit, 
        block_size,
        fp_cols_num,
        compute_quant_scale,
        add_quant_noise_predict
    ):
        self.add_quant_noise = True
        self.add_quant_noise_predict = add_quant_noise_predict
        quant_cols_num = 1
        if block_size > 0:
            quant_cols_num = self.weight.shape[1] - fp_cols_num
            quant_cols_num = int(quant_cols_num // block_size)

        self.quantizer = NoiseQuant(
            weight_shape=self.weight.shape,
            bit=bit,
            block_size=block_size,
            quant_cols_num=quant_cols_num,
            mask=self.mask
        )

        if compute_quant_scale:
            self.quantizer.compute_quant_scale(self.weight)

    def add_quant_bitnoise_to_weight(
        self, 
        bit, 
        block_size,
        fp_cols_num,
        compute_quant_scale,
        add_quant_noise_predict
    ):
        
        self.add_quant_bitnoise = True
        self.add_quant_bitnoise_predict = add_quant_noise_predict
        quant_cols_num = 1
        if block_size > 0:
            quant_cols_num = self.weight.shape[1] - fp_cols_num
            quant_cols_num = int(quant_cols_num // block_size)

        self.quantizer = BitNoiseQuant(
            weight_shape=self.weight.shape,
            bit=bit,
            block_size=block_size,
            quant_cols_num=quant_cols_num,
            mask=self.mask
        )

        if compute_quant_scale:
            self.quantizer.compute_alpha_scale(self.weight)
        
    def quantize_weight(self):
        assert self.quantizer is not None, 'quantizer is not defined!'
        q = self.quantizer.quantize_weight(self.weight)
        self.weight.data = q

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # if self.training:
        #     quantized_weight = self.quantizer(self.weight)
        #     self.weight.data = quantized_weight
        #     return F.linear(input, quantized_weight, self.bias)
        # else:
        #     return F.linear(input, self.weight, self.bias)
        device = self.weight.device
        if self.training:
            if self.training_mode == 'train_full':
                w = self.weight
                if self.learnable_scales:
                    quantized_weight = self.quantizer(w)
                    self.weight.data = quantized_weight
                    return F.linear(input, quantized_weight, self.bias)

            elif self.training_mode == 'train_outlier':
                if self.mask.device != device:
                    self.mask = self.mask.to(device)

                int_weight = self.weight.clone().detach()
                w = torch.where(self.mask, int_weight, self.weight)

            elif self.training_mode == 'train_quant':
                fp_weight = self.weight.clone().detach()
                w = torch.where(self.mask, self.weight, fp_weight)            

            if self.add_quant_noise:
                w = self.quantizer(w)
            
            if self.add_quant_bitnoise:
                w = self.quantizer(w)

            return F.linear(input, w, self.bias)

        else:
            w = self.weight

            if self.add_quant_noise_predict:
                w = self.quantizer(w)
            
            if self.add_quant_bitnoise:
                w = self.quantizer(w)

            return F.linear(input, w, self.bias)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.clip_softmax_eta = config.clip_softmax_eta
        self.clip_softmax_gamma = config.clip_softmax_gamma

        self.enable_clip = False

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        clipped_attn_weights = clipped_softmax(attn_weights, dim=-1, gamma=self.clip_softmax_gamma, eta=self.clip_softmax_eta, dtype=torch.float32).to(query_states.dtype)
        #regular_attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        #if not (torch.isclose(clipped_attn_weights, regular_attn_weights).all()):
         #   print('not consistent')
        attn_weights = nn.functional.dropout(clipped_attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaFlashAttention2(LlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # LlamaFlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class LlamaSdpaAttention(LlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


# LLAMA_ATTENTION_CLASSES = {
#     "eager": LlamaAttention,
#     "flash_attention_2": LlamaAttention, #LlamaFlashAttention2,
#     "sdpa": LlamaAttention,
# }

LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.STE = config.STE
        self.QuantizedLinear_decoder = {
            'replace': config.QuantizedLinear['replace'],
            'outlier_ids': config.QuantizedLinear['outlier_ids'].get(str(layer_idx)),
            'training_mode': config.QuantizedLinear['training_mode']
        }
        self.weight_quant_noise_decoder = {
            'add': config.weight_quant_noise['add'],
            'predict': config.weight_quant_noise['predict'],
            'compute_scale': config.weight_quant_noise['compute_scale'],
            'layer_bit': config.weight_quant_noise['layer_bit'].get(str(layer_idx)),
            'block_size': config.weight_quant_noise['block_size'],
            'fp_cols_num': config.weight_quant_noise['fp_cols_num']
        }

        self.weight_quant_bitnoise_decoder = {
            'add': config.weight_quant_bitnoise['add'],
            'predict': config.weight_quant_bitnoise['predict'],
            'compute_scale': config.weight_quant_bitnoise['compute_scale'],
            'layer_bit': config.weight_quant_bitnoise['layer_bit'].get(str(layer_idx)),
            'block_size': config.weight_quant_bitnoise['block_size'],
            'fp_cols_num': config.weight_quant_bitnoise['fp_cols_num']
        }

        if self.QuantizedLinear_decoder['replace']:
            self.replace_Linear()
        
        if self.weight_quant_noise_decoder['add']:
            self.add_quant_noise_to_weight()

        if self.weight_quant_bitnoise_decoder['add']:
            self.add_quant_bitnoise_to_weight()
        

        # self.add_quant_noise = config.add_quant_noise
        # self.add_quant_noise_predict = config.add_quant_noise_predict
        # self.training_mode = config.training_mode
        # self.layer_bit = config.layer_bit.get(layer_idx)
        # self.block_size = config.block_size
        # self.outlier_ids = config.outlier_ids.get(layer_idx)
        # self.learnable_scales = config.learnable_scales

        # self.outlier_ids = config.outlier_ids.get(layer_idx)


        # self.STE = config.STE
        # if self.STE:
        #     self.layer_bit = config.layer_bit[layer_idx]       
        #     self.block_size = config.block_size
        #     # self.outlier_ids = config.outlier_ids[layer_idx]
        #     self.learnable_scales = config.learnable_scales
        
        # self.add_quant_noise = config.add_quant_noise
        # if self.add_quant_noise:
        #     self.layer_bit = config.layer_bit[layer_idx]
        #     # self.outlier_ids = config.outlier_ids[layer_idx]
        #     self.block_size = config.block_size
        #     self.learnable_scales = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        if self.STE and self.training and (not self.learnable_scales):
            self.quantize()

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    def quantize(self):
        for layer_name in self.outlier_ids.keys():
            cur_layer = attrgetter(layer_name)(self)
            cur_layer.weight.data = quantize_with_outliers(cur_layer.weight.data, B=self.layer_bit[layer_name], block_size=self.block_size, idx=self.outlier_ids[layer_name])

    def replace_Linear(self): 
        # self.learnable_scales = True

        layers = ['self_attn', 'mlp']
        projectors = {
            'self_attn': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            'mlp': ['up_proj', 'down_proj', 'gate_proj']
        }
        for layer_name in layers:
            cur_layer = getattr(self, layer_name)
            for proj_name in projectors[layer_name]:
                cur_projection = getattr(cur_layer, proj_name)
                
                outlier_ids = self.QuantizedLinear_decoder['outlier_ids'].get(f'{layer_name}.{proj_name}')
                training_mode = self.QuantizedLinear_decoder['training_mode']

                quantized_projection = QuantizedLinear(
                    cur_projection,
                    training_mode=training_mode
                )

                if outlier_ids is not None:
                    quantized_projection.set_mask(outlier_ids)
                
                setattr(cur_layer, proj_name, quantized_projection)

    def add_quant_noise_to_weight(self):
        layers = ['mlp', 'self_attn']
        projectors = {
            'self_attn': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            'mlp': ['up_proj', 'down_proj', 'gate_proj']
        }
        for layer_name in layers:
            cur_layer = getattr(self, layer_name)
            for proj_name in projectors[layer_name]:
                cur_projection = getattr(cur_layer, proj_name)

                if isinstance(cur_projection, QuantizedLinear):
                    
                    add_noise_to_predict = self.weight_quant_noise_decoder['predict']
                    compute_quant_scale = self.weight_quant_noise_decoder['compute_scale']
                    cur_bit = self.weight_quant_noise_decoder['layer_bit'].get(f'{layer_name}.{proj_name}')
                    block_size = self.weight_quant_noise_decoder['block_size']
                    fp_cols_num = self.weight_quant_noise_decoder['fp_cols_num']
                    
                    cur_projection.add_quant_noise_to_weight(
                        bit=cur_bit, 
                        block_size=block_size,
                        fp_cols_num=fp_cols_num,
                        compute_quant_scale=compute_quant_scale,
                        add_quant_noise_predict=add_noise_to_predict
                    )
        self.weight_quant_noise_decoder['compute_scale'] = False

    def add_quant_bitnoise_to_weight(self):
        layers = ['self_attn', 'mlp']
        projectors = {
            'self_attn': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            'mlp': ['up_proj', 'down_proj', 'gate_proj']
        }
        for layer_name in layers:
            cur_layer = getattr(self, layer_name)
            for proj_name in projectors[layer_name]:
                cur_projection = getattr(cur_layer, proj_name)

                if isinstance(cur_projection, QuantizedLinear):
                    
                    add_noise_to_predict = self.weight_quant_bitnoise_decoder['predict']
                    compute_quant_scale = self.weight_quant_bitnoise_decoder['compute_scale']
                    cur_bit = self.weight_quant_bitnoise_decoder['layer_bit'].get(f'{layer_name}.{proj_name}')
                    block_size = self.weight_quant_bitnoise_decoder['block_size']
                    fp_cols_num = self.weight_quant_bitnoise_decoder['fp_cols_num']
                    
                    cur_projection.add_quant_bitnoise_to_weight(
                        bit=cur_bit, 
                        block_size=block_size,
                        fp_cols_num=fp_cols_num,
                        compute_quant_scale=compute_quant_scale,
                        add_quant_noise_predict=add_noise_to_predict
                    )
        self.weight_quant_bitnoise_decoder['compute_scale'] = False

    def quantize_weight(self):
        layers = ['self_attn', 'mlp']
        projectors = {
            'self_attn': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            'mlp': ['up_proj', 'down_proj', 'gate_proj']
        }
        for layer_name in layers:
            cur_layer = getattr(self, layer_name)
            for proj_name in projectors[layer_name]:
                cur_projection = getattr(cur_layer, proj_name)

                if isinstance(cur_projection, QuantizedLinear):
                    cur_projection.quantize_weight()

LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if not getattr(config, 'clip_softmax_gamma', None):
            config.clip_softmax_gamma = 0

        if not getattr(config, 'clip_softmax_eta', None):
            config.clip_softmax_eta = 1

        # if not getattr(config, 'layer_bit', None):
        #     config.layer_bit = {}
        
        # if not getattr(config, 'block_size', None):
        #     config.block_size = 0

        # if not getattr(config, 'outlier_ids', None):
        #     config.outlier_ids = {}

        if not getattr(config, 'STE', None):
            config.STE = False

        if not getattr(config, 'learnable_scales', None):
            config.learnable_scales = False

        if not getattr(config, 'QuantizedLinear', None):
            config.QuantizedLinear = {
                'replace': False,
                'outlier_ids': {},
                'training_mode': 'train_full'
            }
        
        if not getattr(config, 'weight_quant_noise', None):
            config.weight_quant_noise = {
                'add': False,
                'predict': False,
                'compute_scale': False,
                'layer_bit': {},
                'block_size': None,
                'fp_cols_num': None
            }

        if not getattr(config, 'weight_quant_bitnoise', None):
            config.weight_quant_bitnoise = {
                'add': False,
                'predict': False,
                'compute_scale': False,
                'layer_bit': {},
                'block_size': None,
                'fp_cols_num': None
            }
        
        # if not getattr(config, 'add_quant_noise', None):
        #     config.add_quant_noise = False        
        
        # if not getattr(config, 'add_quant_noise_predict', None):
        #     config.add_quant_noise_predict = False
        
        # if not getattr(config, 'add_quant_noise_predict', None):
        #     config.compute_quant_scales = False
        


        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value


    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def set_clipped_sm(self, gamma, eta):
        for layer in self.model.layers:
            layer.self_attn.clip_softmax_eta = eta
            layer.self_attn.clip_softmax_gamma = gamma
        self.model.config.clip_softmax_gamma = gamma
        self.model.config.clip_softmax_eta = eta

    # def enable_ste(self, outlier_ids, layer_bit, block_size=64, learnable_scales=False):
    #     self.model.config.outlier_ids = outlier_ids
    #     self.model.config.layer_bit = layer_bit
    #     self.model.config.block_size = block_size
    #     self.model.config.STE = True
    #     self.model.config.learnable_scales = learnable_scales
    #     self.model.config.add_quant_noise = False

    #     for layer_idx in range(self.model.config.num_hidden_layers):
    #         self.model.layers[layer_idx].layer_bit = layer_bit[layer_idx]
    #         self.model.layers[layer_idx].outlier_ids = outlier_ids[layer_idx]
    #         self.model.layers[layer_idx].block_size = block_size
    #         self.model.layers[layer_idx].STE = True
    #         self.model.layers[layer_idx].add_quant_noise = False
    #         if learnable_scales:
    #             self.model.layers[layer_idx].learnable_scales = True
    #             self.model.layers[layer_idx].replace_Linear()
    
    def replace_Linear(self, outlier_ids, training_mode='train_outlier'):
        self.model.config.QuantizedLinear['replace'] = True
        self.model.config.QuantizedLinear['outlier_ids'] = outlier_ids
        self.model.config.QuantizedLinear['training_mode'] = training_mode

        for layer_idx in range(self.model.config.num_hidden_layers):
                self.model.layers[layer_idx].QuantizedLinear_decoder = {
                    'replace': self.model.config.QuantizedLinear['replace'],
                    'outlier_ids': self.model.config.QuantizedLinear['outlier_ids'].get(str(layer_idx)),
                    'training_mode': self.model.config.QuantizedLinear['training_mode']
                }
                self.model.layers[layer_idx].replace_Linear()

    def add_quant_noise_to_weight(self, layer_bit, block_size=128, fp_cols_num=128, compute_scale = True, quant_noise_predict=False):
        self.model.config.weight_quant_noise['add'] = True
        self.model.config.weight_quant_noise['predict'] = quant_noise_predict
        self.model.config.weight_quant_noise['compute_scale'] = compute_scale
        self.model.config.weight_quant_noise['layer_bit'] = layer_bit
        self.model.config.weight_quant_noise['block_size'] = block_size
        self.model.config.weight_quant_noise['fp_cols_num'] = fp_cols_num

        for layer_idx in range(self.model.config.num_hidden_layers):
            self.model.layers[layer_idx].weight_quant_noise_decoder = {
                'add': self.model.config.weight_quant_noise['add'],
                'predict': self.model.config.weight_quant_noise['predict'],
                'compute_scale': self.model.config.weight_quant_noise['compute_scale'],
                'layer_bit': self.model.config.weight_quant_noise['layer_bit'].get(str(layer_idx)),
                'block_size': self.model.config.weight_quant_noise['block_size'],
                'fp_cols_num': self.model.config.weight_quant_noise['fp_cols_num']
            }
            self.model.layers[layer_idx].add_quant_noise_to_weight()
        
        self.model.config.weight_quant_noise['compute_scale'] = False


    def add_quant_bitnoise_to_weight(self, layer_bit, block_size=128, fp_cols_num=128, compute_scale = True, quant_noise_predict=False):
        self.model.config.weight_quant_bitnoise['add'] = True
        self.model.config.weight_quant_bitnoise['predict'] = quant_noise_predict
        self.model.config.weight_quant_bitnoise['compute_scale'] = compute_scale
        self.model.config.weight_quant_bitnoise['layer_bit'] = layer_bit
        self.model.config.weight_quant_bitnoise['block_size'] = block_size
        self.model.config.weight_quant_bitnoise['fp_cols_num'] = fp_cols_num

        for layer_idx in range(self.model.config.num_hidden_layers):
            self.model.layers[layer_idx].weight_quant_bitnoise_decoder = {
                'add': self.model.config.weight_quant_bitnoise['add'],
                'predict': self.model.config.weight_quant_bitnoise['predict'],
                'compute_scale': self.model.config.weight_quant_bitnoise['compute_scale'],
                'layer_bit': self.model.config.weight_quant_bitnoise['layer_bit'].get(str(layer_idx)),
                'block_size': self.model.config.weight_quant_bitnoise['block_size'],
                'fp_cols_num': self.model.config.weight_quant_bitnoise['fp_cols_num']
            }
            self.model.layers[layer_idx].add_quant_bitnoise_to_weight()
        
        self.model.config.weight_quant_bitnoise['compute_scale'] = False

    def quantize_weight(self):
        for layer_idx in range(self.model.config.num_hidden_layers):
            self.model.layers[layer_idx].quantize_weight()
    
    # def change_training_mode(self, outlier_ids, training_mode):
    #     assert training_mode in ['train_full', 'train_outlier', 'train_quant'], 'Incorrect mode!'
        
    #     self.model.config.training_mode = training_mode
    #     for layer_idx in range(self.model.config.num_hidden_layers):
    #         self.model.layers[layer_idx].outlier_ids = outlier_ids[layer_idx]
    #         self.model.layers[layer_idx].training_mode = training_mode
    #         self.model.layers[layer_idx].replace_Linear()


    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
