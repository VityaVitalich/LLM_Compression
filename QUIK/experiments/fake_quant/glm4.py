import os

import argparse
import datautils
import modelutils
import torch
import quik_utils
import quant
import sparseGPT_utils
import types
import torch.nn.functional as F
DEV = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
funcType = types.MethodType

def unquantized_forward(self, input: torch.Tensor) -> torch.Tensor:
    return F.linear(input, self.weight, self.bias)

def glm_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--model', type=str,
        help='GLM model to load;',
    )
    parser.add_argument(
        '--seq-len', type=int,
        help='Length of sequence',
    )

    parser.add_argument(
        '--replace_QuantizedLinear',
        help='Replace QuantizedLinear layers by Linear ones to quantize model',
        action='store_true'
    )

    parser.add_argument(
        '--path_to_act_scales', type=str,
        help='act_scales to load;',
        default='../act_scales/Llama-2-7b-hf.pt'
    )

    parser.add_argument(
        '--path_to_save_quant_model', type=str,
        help='path to save model after quantization;'
    )    

    parser.add_argument(
        '--dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.', default='c4'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument('--fp_features', type=int, default=0, help='Number of features to keep in FP16.')
    parser.add_argument('--fp_threshold', type=float, default=0.0, help='Threshold where we put the fp features to zero.')
    parser.add_argument('--fp_relative', action='store_true', help='Use relative features for number of fp_features (larger layers have more fp_features)')
    # parser.add_argument('--fp_relative', type=str, help='Use relative features for number of fp_features (larger layers have more fp_features)')
    # Act. Quantization Params:
    parser.add_argument('--a_bits', type=int, default=16, choices=[4, 8, 16])

    # Weight Quantization Params: 
    parser.add_argument('--w_bits', type=int, default=16, choices=[2, 3, 4, 8, 16])
    parser.add_argument('--w_clip', action='store_true', help='Use clipping for weight quantization')
    parser.add_argument('--w_asym', action='store_true')
    
    parser.add_argument('--int8_down_proj', action='store_true', help='Use INT8 for Down Projection')
    
    # SparseGPT arguments:
    parser.add_argument('--sparsity', type=float, default=0, help='Target sparsity')
    parser.add_argument('--prunen', type=int, default=0,help='N for N:M pruning.')
    parser.add_argument('--prunem', type=int, default=0,help='M for N:M pruning.')    
    
    # Wandb Args:
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_name', type=str, default='anonymous')

    parser.add_argument('--int8_2_4', action='store_true', help='Use SparseGPT int8 2:4 quantization with SmoothQuant')
    parser.add_argument('--smoothquant', action='store_true', help='Use SmoothQuant Baseline')

    parser.add_argument('--synthetic_data', action='store_true', help='Use Synthetic Data (for debugging purposes)')
    parser.add_argument('--hf_token', type=str, default='')
    
    args = parser.parse_args()
    if args.smoothquant or args.int8_2_4:
        assert args.smoothquant and args.int8_2_4 == False, 'Cannot use both SmoothQuant and SparseGPT int8 2:4 quantization!'
    if args.sparsity >0 or args.prunen + args.prunem > 0:
        args.sparseGPT = True
    else:
        args.sparseGPT = False
    return args



@torch.no_grad()
def glm_sequential(model, dataloader, act_scales, dev, args):
    print('Starting ...')

    if getattr(model.config, 'use_cache', False):
        use_cache = model.config.use_cache
        model.config.use_cache = False

    layers = model.transformer.encoder.layers
    save_dict = {}

    #model.model.embed_tokens = model.model.embed_tokens.to(dev)
    #model.model.norm = model.model.norm.to(dev)
    model.transformer.embedding.to(dev)
    model.transformer.rotary_pos_emb.to(dev)
    model.transformer.output_layer.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, args.seq_len, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, hidden_states, attention_mask, rotary_pos_emb, *args, **kwargs):
            inps[cache['i']] = hidden_states
            cache['i'] += 1
            cache['attention_mask'] = attention_mask
            cache['rotary_pos_emb'] = rotary_pos_emb
           # cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(input_ids=batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
  #  model.model.embed_tokens = model.model.embed_tokens.cpu()
   # model.model.norm = model.model.norm.cpu()
    
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    rotary_pos_emb = cache['rotary_pos_emb']

    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        print(f'\nLayer {i}:', flush=True, end=' ')
        layer = layers[i].to(dev)
        full = modelutils.find_layers(layer)
        sequential = [
            ['self_attention.query_key_value'],
            ['self_attention.dense'],
            ['mlp.dense_h_to_4h'],
            ['mlp.dense_4h_to_h']
        ]
        for names in sequential:
            subset = {n: full[n] for n in names}
            
            modules_quik = {}
            for name in subset:
                print(f'{name}', end='  ', flush=True)   
                
                # Extract the number of outliers
                if args.fp_relative:
                    raise NotImplementedError
                else:
                    outlier_num = args.fp_features
                
                layer_scales = None
                if outlier_num > 0:
                    layer_scales = act_scales['transformer.encoder.layers.{}.{}'.format(i, name)]

                    max_val = layer_scales.abs().max()
                    fp_threshold = args.fp_threshold

                    if 'down_proj' in name and args.int8_down_proj:
                        fp_threshold *= 2
                    
                    if max_val <= fp_threshold:
                        outlier_num = 0
                        layer_scales = None

                if args.sparseGPT:
                    raise NotImplementedError
                else:
                    modules_quik[name] = quik_utils.QUIK(
                    layer=subset[name],
                    act_scales=layer_scales,
                    fp_features=outlier_num
                    )
                modules_quik[name].quantizer = quant.WeightQuantizer()

                current_w_bits = args.w_bits 
                if 'down_proj' in name:
                    if args.int8_down_proj:
                        raise NotImplementedError

                ste_scales = None
                modules_quik[name].quantizer.configure(
                    current_w_bits, perchannel=True, sym=not(args.w_asym), mse=args.w_clip, scales=ste_scales
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    modules_quik[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask, rotary_pos_emb)[0]
            for h in handles:
                h.remove()

            for name in subset:
                if args.sparseGPT:
                    raise NotImplementedError
                else:
                    modules_quik[name].fasterquant(percdamp=args.percdamp, groupsize=-1)
                
                quantizers['glm.transformer.layers.%d.%s' % (i, name)] = modules_quik[name].quantizer
                save_dict['glm.transformer.layers.%d.%s' % (i, name)] = {}
                


                save_dict['glm.transformer.layers.%d.%s' % (i, name)]['alpha'] = modules_quik[name].quantizer.alpha.to("cpu")

                
                modules_quik[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask, rotary_pos_emb)[0]
        layers[i] = layer.cpu()
        del layer
        del modules_quik 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    if getattr(model.config, 'use_cache', False):
        model.config.use_cache = use_cache
    
    return quantizers, save_dict


if __name__ == '__main__':
    args = glm_parser()
    datautils.set_seed(args.seed)
    
    print(args)
    if args.wandb:
        import wandb
        wandb.init(project="quik", entity=args.wandb_name)
        wandb.config.update(args)
    
    model = modelutils.get_glm(args.model, args.hf_token)
    model.eval()
    
    # Extract Scale
    if args.w_bits < 16 or args.a_bits < 16 or args.int8_2_4 or args.smoothquant or args.sparseGPT:
        if args.fp_features > 0 or args.int8_2_4 or args.smoothquant:
            relative_path = args.path_to_act_scales
            act_scales = torch.load(relative_path)
            print('Loaded act_scales from: ', relative_path)
        else:
            act_scales = None
            print('No act_scales loaded')
    if args.int8_2_4:
        raise NotImplementedError   
    elif args.smoothquant:
        raise NotImplementedError
    # Apply GPTQ on the model
    elif args.w_bits < 16 or args.sparseGPT:
        dataloader, testloader = datautils.get_loaders(
            args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=args.seq_len, 
            synthetic_data=args.synthetic_data, hf_token=args.hf_token
        )
        quantizers, save_dict = glm_sequential(model, dataloader, act_scales, DEV, args)
    
    # Add Input Quantization
    if args.a_bits < 16:
        raise NotImplementedError

    save_path = args.path_to_save_quant_model
    model.save_pretrained(save_path)
    # No eval for GLM
    # datasets = ['wikitext2']
    # for dataset in datasets:
    #     dataloader, testloader = datautils.get_loaders(
    #         dataset, seed=args.seed, model=args.model, seqlen=model.seqlen, hf_token=args.hf_token
    #     )
    #     print(dataset)
    #     dataset_ppl = modelutils.llama_eval(model, testloader, DEV)
    #     print(f'\n{dataset.upper()} PPL: {dataset_ppl:.3f}')
    #     print(40*'-')
    #     if args.wandb:
    #         wandb.log({'ppl/{}'.format(dataset): dataset_ppl})
        
