# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
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

import math
import warnings
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D
#from transformers.models.llama.modeling_llama import QuantizedLinear

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.other import transpose

from .config import SASUTConfig


class LoRALayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("sasut_noise")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ()

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})

        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, QuantizedLinear):
            if base_layer.outlier_cols_num == 0:
                in_features, out_features = base_layer.in_features, base_layer.out_features
            elif base_layer.outlier_cols_num > 0:
                in_features, out_features = base_layer.q_in_features, base_layer.out_features
                self.quant_mask = base_layer.mask

        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features
        
        if self.kwargs.get("quant_noise_config"):
            self.add_quant_noize = True
            quant_noise_config = self.kwargs["quant_noise_config"]
            self.quant_bit = quant_noise_config["quant_bit"]
            self.quant_block_size = quant_noise_config["quant_block_size"]
            self.outliers_fraction = quant_noise_config["outliers_fraction"]
            self.quant_range_params = self.get_quant_range_params(
                self.base_layer, 
                self.quant_block_size
            )
        else:
            self.add_quant_noize = False
        
        if self.kwargs.get("quant_mask_config"):
            quant_mask_config = self.kwargs["quant_mask_config"]
            self.add_quant_mask = quant_mask_config["add_quant_mask"]
            self.outlier_cols_num = quant_mask_config["outlier_cols_num"]
            self.path_to_outlier_ids = quant_mask_config["path_to_outlier_ids"]
            self.quant_mask = nn.ParameterDict({})
            
            # self.base_layer.weight.requires_grad = True



    def get_outliers_mask(self, weight, outlier_fraction):
        with torch.no_grad():
            w = weight
            w_flat = w.view(-1).clone().float()
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

        outliers_mask = outliers.detach()

        return outliers_mask

    def get_quant_range_params(
        self,
        layer_in,
        block_size = None
    ):  
        weight_in = layer_in.weight.detach().clone()
        cols_num = layer_in.in_features

        if self.outliers_fraction > 0:
            outliers_mask = self.get_outliers_mask(weight_in, self.outliers_fraction)
            weight_mask = ~outliers_mask
            weight_in = weight_mask * weight_in


        if (block_size is None) or (block_size == 0) or (block_size >= cols_num):
            range_params = weight_in.max(dim=0)[0] - weight_in.min(dim=0)[0]

        else:
            range_params = torch.ones(cols_num, dtype=weight_in.dtype)
            for i in range(0, cols_num, block_size):
                min_val = weight_in[:, i:(i + block_size)].min()
                max_val = weight_in[:, i:(i + block_size)].max()
                block_range = (max_val - min_val)
                range_params[i:(i + block_size)] = block_range * range_params[i:(i + block_size)]

        range_params = range_params.detach()
        return range_params

    def get_quant_mask(self, adapter_name):
        self.quant_mask[adapter_name] = torch.ones(self.in_features, 
                                                   dtype=torch.bool)
        # self.path_to_outlier_ids
        



    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        if self.kwargs.get("quant_mask_config"):
            if self.add_quant_mask:
                self.get_mask(adapter_name)

        # check weight and qweight (for GPTQ)
        for weight_name in ("weight", "qweight"):
            weight = getattr(self.get_base_layer(), weight_name, None)
            if weight is not None:
                # the layer is already completely initialized, this is an update
                if weight.dtype.is_floating_point or weight.dtype.is_complex:
                    self.to(weight.device, dtype=weight.dtype)
                else:
                    self.to(weight.device)
                break
        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        if adapter_name in self.lora_A.keys():
            if init_lora_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            elif init_lora_weights.lower() == "gaussian":
                nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_lora_weights=}")
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])

    def loftq_init(self, adapter_name):
        from peft.utils.loftq_utils import loftq_init

        weight = self.get_base_layer().weight
        kwargs = {
            "num_bits": self.kwargs.get("loftq_bits", 4),
            "reduced_rank": self.r[adapter_name],
            "num_iter": self.kwargs.get("loftq_iter", 1),
        }

        qweight, lora_A, lora_B = loftq_init(weight, **kwargs)
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            self.lora_A[adapter_name].weight.data = lora_A
            self.lora_B[adapter_name].weight.data = lora_B
        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            self.lora_embedding_A[adapter_name].weight.data = lora_A
            self.lora_embedding_B[adapter_name].weight.data = lora_B
        self.get_base_layer().weight.data = qweight

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return
        self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            self.scaling[active_adapter] *= scale

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            if scale is None:
                self.scaling[active_adapter] = self.lora_alpha[active_adapter] / self.r[active_adapter]
            else:
                self.scaling[active_adapter] /= scale


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class Linear(nn.Module, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )

        if adapter_names is None:
            adapter_names = self.active_adapters

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]                
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                
                if x.dtype != lora_A.weight.dtype:
                    x = x.to(lora_A.weight.dtype)
                if hasattr(self, "quant_mask"):
                    x = x[:, :, self.quant_mask]
                result += lora_B(lora_A(dropout(x))) * scaling

                # if self.add_quant_noize:
                #     device = lora_A.weight.device
                #     weight_randn = self.quant_noise(
                #         self.base_layer, 
                #         self.quant_range_params, 
                #         self.quant_bit, 
                #         device
                #     )
                #     result += F.linear(x, weight_randn)

        result = result.to(previous_dtype)
        return result

    def quant_noise(self, layer_in, range_params, quant_bit, device):
        weight_in = layer_in.weight
        weight_randn = torch.randn_like(
            weight_in, requires_grad=False, device=device) / 2
        
        if range_params.device != device:
            range_params = range_params.to(device)
        
        unit = 1 / (2**quant_bit - 1)
        weight_randn = range_params * unit * weight_randn

        return weight_randn

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep

class QuantizedLinear(nn.Module):
    def __init__(
        self,
        target,
        adapter_name,
        sasut_config,
        outlier_ids
    ):
        super(QuantizedLinear, self).__init__()

        self.in_features = target.in_features
        self.out_features = target,out_features
        self.q_in_features = None
        self.fp_in_features = None
        self.outlier_cols_num = sasut_config['outlier_num']
        self.device = None
        self.dtype = dtype
        self.is_quant_weight = False
        self.training_mode = None
        self.quantizer = None
        self.noisemaker = None
        self.add_quant_bitnoise = None
        self.add_quant_bitnoise_predict = None
        self.outlier_ids = outlier_ids

        if self.outlier_cols_num == 0:
            raise ValueError('outlier col num is 0')

        elif self.outlier_cols_num > 0:
            self.q_in_features = self.in_features - self.outlier_cols_num
            self.fp_in_features = self.outlier_cols_num
            
            q_weight = torch.rand((self.out_features, self.q_in_features), 
                                    dtype=self.dtype)
            fp_weight = torch.rand((self.out_features, self.fp_in_features), 
                                    dtype=self.dtype)

            self.q_weight = nn.Parameter(q_weight)
            self.fp_weight = nn.Parameter(fp_weight)
            self.bias = None

            mask = torch.ones(self.in_features, 
                              dtype=torch.bool)
            col_perm = torch.arange(self.in_features, 
                                    dtype=torch.int32)
            inv_col_perm = torch.zeros(col_perm.numel(), 
                                       dtype=col_perm.dtype)           

            self.register_buffer("mask", mask)
            self.register_buffer("col_perm", col_perm)
            self.register_buffer("inv_col_perm", inv_col_perm)

        else:
            raise ValueError('Number of outlier columns should be non-negative!')

        self.from_fp_Linear(target, outlier_ids)
    

    @torch.no_grad
    def set_mask(self, outlier_ids: torch.tensor):
        self.mask = torch.ones(self.in_features, 
                               dtype=torch.bool)
        self.mask[outlier_ids] = False

        col_ids = torch.arange(self.in_features, 
                               dtype=torch.int32)
        self.col_perm = torch.cat([col_ids[self.mask], 
                                   col_ids[~self.mask]])

        self.inv_col_perm = torch.zeros(self.col_perm.numel(), 
                                        dtype=self.col_perm.dtype)
        self.inv_col_perm[self.col_perm] = torch.arange(self.col_perm.numel(),
                                                        dtype=self.col_perm.dtype)       

    def from_fp_Linear(
        self,
        linear: torch.nn.Linear,
        outlier_ids: List = None,
    ):
        if self.outlier_cols_num == 0:            
            self.weight.data = linear.weight.data
            self.bias = nn.Parameter(linear.bias.data) if linear.bias is not None else None
        
        elif self.outlier_cols_num > 0:
            if len(outlier_ids) > 0:
                self.set_mask(outlier_ids)
    
            weight = linear.weight.data
            self.q_weight.data = weight[:, self.mask]
            self.fp_weight.data = weight[:, ~self.mask]

            self.bias = nn.Parameter(linear.bias.data) if linear.bias is not None else None

           # self.set_training_mode(training_mode)
           # this is implemented in _mark_only_adapters_as_trainable function
    
    # def set_training_mode(
    #     self, 
    #     training_mode: Literal['train_full', 'train_outlier', 'train_quant']
    # ):
    #     self.training_mode = training_mode

    #     if self.training_mode == 'train_full':
    #         self.q_weight.requires_grad = True 
    #         self.fp_weight.requires_grad = True 

    #     elif self.training_mode == 'train_outlier':
    #         self.q_weight.requires_grad = False
    #         self.fp_weight.requires_grad = True
    
    #     elif self.training_mode == 'train_quant':
    #         self.q_weight.requires_grad = True
    #         self.fp_weight.requires_grad = False

    def add_symmetric_quantizer(
        self,
        bit: int,
        learnable_scale: bool = False
    ):
                
        self.quantizer = SymQuant(
            out_features=self.out_features,
            bit=bit    
        )
        if learnable_scale:
            self.quantizer.alpha_scale.requires_grad = True
        else:
            self.quantizer.alpha_scale.requires_grad = False

    def add_quant_weight(
        self,
        alpha_scale: torch.tensor,
        quant_weight: torch.tensor,
        fp_weight: torch.tensor = None
    ):  
        self.is_quant_weight = True

        if fp_weight is not None:
            assert self.mask is not None, \
                'self.mask is None, try using set_mask method'

            quant_weight = quant_weight.to(dtype=self.dtype, 
                                           device=self.device)

            quant_weight = quant_weight[:, self.mask]
            self.q_weight.data = quant_weight
            self.fp_weight.data = fp_weight.to(dtype=self.dtype, 
                                               device=self.device)
            
        else:
            self.weight = quant_weight.to(dtype=self.dtype, 
                                          device=self.device)

        self.quantizer.alpha_scale.data = alpha_scale.to(dtype=self.dtype, 
                                                         device=self.device)

    def add_quant_bitnoise_to_weight(
        self, 
        bit,
        noise_type,
        compute_quant_scale,
        add_quant_noise_predict
    ):
        
        self.add_quant_bitnoise = True
        self.add_quant_bitnoise_predict = add_quant_noise_predict

        if self.outlier_cols_num == 0:
            w = self.weight
        elif self.outlier_cols_num > 0:
            w = self.q_weight

        self.noisemaker = BitNoiseQuant(
            out_features=w.shape[0],
            bit=bit,
            noise_type=noise_type
        )

        if compute_quant_scale:
            if self.is_quant_weight:
                w = self.quantizer.dequantize(w)
            self.noisemaker.compute_alpha_scale(w)

    def _noise_cond(self):
        noise_cond1 = (self.add_quant_bitnoise and self.training)
        noise_cond2 = (self.add_quant_bitnoise_predict and (not self.training))
        noise_cond = noise_cond1 or noise_cond2

        return noise_cond       

    def forward_with_quant_weight(self, input):
        noise_cond = self._noise_cond()
        
        if self.outlier_cols_num == 0:
            dq_w = self.quantizer.dequantize(self.weight)
            if noise_cond:
               dq_w = self.noisemaker(dq_w) 

            return F.linear(input, dq_w, self.bias)

        elif self.outlier_cols_num > 0:
            dq_w = self.quantizer.dequantize(self.q_weight)
            if noise_cond:
               dq_w = self.noisemaker(dq_w) 

            out_w = torch.hstack([dq_w,
                                  self.fp_weight])
            out_w = out_w[:, self.inv_col_perm]

            return F.linear(input, out_w, self.bias)

    def forward_with_fp_weight(self, input):
        noise_cond = self._noise_cond()

        if self.outlier_cols_num == 0:
            w = self.weight
            if noise_cond:
                w = self.noisemaker(w)

            return F.linear(input, w, self.bias)
        
        elif self.outlier_cols_num > 0:

            if noise_cond:
                quant_w_noised = self.noisemaker(self.q_weight)
                out_w = torch.hstack([quant_w_noised, 
                                      self.fp_weight])               
            else:
                out_w = torch.hstack([self.q_weight, 
                                      self.fp_weight])

            out_w = out_w[:, self.inv_col_perm]

            return F.linear(input, out_w, self.bias)   

    def forward(self, input):
        if self.is_quant_weight:
            out = self.forward_with_quant_weight(input)
            return out
        else:
            out = self.forward_with_fp_weight(input)
            return out

def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    sasut_config: SASUTConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    # if isinstance(target_base_layer, torch.nn.Embedding):
    #     embedding_kwargs = kwargs.copy()
    #     embedding_kwargs.pop("fan_in_fan_out", None)
    #     embedding_kwargs.update(lora_config.loftq_config)
    #     new_module = Embedding(target, adapter_name, **embedding_kwargs)
    # elif isinstance(target_base_layer, torch.nn.Conv2d):
    #     kwargs.update(lora_config.loftq_config)
    #     new_module = Conv2d(target, adapter_name, **kwargs)
    if isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            raise NotImplementedError('no fan in fan out')
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
       # kwargs.update(lora_config.loftq_config)
        new_module = QuantizedLinear(target, adapter_name, sasut_config)
    # elif isinstance(target_base_layer, QuantizedLinear):
    #     if kwargs["fan_in_fan_out"]:
    #         warnings.warn(
    #             "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
    #             "Setting fan_in_fan_out to False."
    #         )
    #         kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
    #     kwargs.update(lora_config.loftq_config)
    #     new_module = Linear(target, adapter_name, **kwargs)
    # elif isinstance(target_base_layer, Conv1D):
    #     if not kwargs["fan_in_fan_out"]:
    #         warnings.warn(
    #             "fan_in_fan_out is set to False but the target module is `Conv1D`. " "Setting fan_in_fan_out to True."
    #         )
    #         kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
    #     kwargs.update(lora_config.loftq_config)
    #     new_module = Linear(target, adapter_name, is_target_conv_1d_layer=True, **kwargs)

    return new_module
