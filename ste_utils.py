import torch
import re


def get_fp_inds_for_quik(path_to_act_scales, fp_features_num): 
    act_scales = torch.load(path_to_act_scales) 
    fp_indices_in_lin_layers = {k: torch.sort(v)[1][-fp_features_num:] for k, v in act_scales.items()} 
    return fp_indices_in_lin_layers

def extract_pattern(s):
    pattern = r'model\.layers\.(\d+)\.(.+)'

    # Perform regex search
    match = re.search(pattern, s)

    layer_number = match.group(1)
    rest_of_string = match.group(2)
    return layer_number, rest_of_string

def get_fp_llama(path_to_act_scales, fp_features_num):
    fp_indices_in_lin_layers = get_fp_inds_for_quik(path_to_act_scales, fp_features_num)

    outlier_ids = {}
    for k, v in fp_indices_in_lin_layers.items():

        if not 'layer' in  k:
            continue 

        layer_number, rest_of_string = extract_pattern(k)
        
        if int(layer_number) not in outlier_ids.keys():
            outlier_ids[int(layer_number)] = {rest_of_string: v}
        else:
            outlier_ids[int(layer_number)][rest_of_string] = v

    return outlier_ids


def make_layer_bits(outlier_ids, q=4, k=4, v=4, o=4, down=4, gate=4, up=4):
    layer_bits = {}
    for layer_num in outlier_ids.keys():
        layer_bits[layer_num] = {}
        layer_bits[layer_num]['self_attn.q_proj'] = q
        layer_bits[layer_num]['self_attn.k_proj'] = k
        layer_bits[layer_num]['self_attn.v_proj'] = v
        layer_bits[layer_num]['self_attn.o_proj'] = o
        layer_bits[layer_num]['mlp.gate_proj'] = gate
        layer_bits[layer_num]['mlp.up_proj'] = up
        layer_bits[layer_num]['mlp.down_proj'] = down

    return layer_bits

def prepare_llama_ste(path_to_act_scales, fp_features_num, **kwargs):
    outlier_ids = get_fp_llama(path_to_act_scales, fp_features_num)
    layer_bits = make_layer_bits(outlier_ids, **kwargs)

    return outlier_ids, layer_bits