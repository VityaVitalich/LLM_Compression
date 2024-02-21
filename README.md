# QUIK
This repository contains the code to fake quantize LLaMa2 model with QUIK and perform knowledge recovery of the quantized model by fine-tuning its columns with outliers.

The method was tested for **3-bit** quantization.

## Install

### Dependencies

- python 3.10.13
- pytorch 2.1.0
- cuda12.1
- cudnn8

The tests were preformed using the pytorch docker image:
```bash
docker pull pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
```

### Instructions

```bash
git clone -b export https://github.com/VityaVitalich/LLM_Compression.git

cd LLM_compression/transformers_modified
pip install .

pip install sentencepiece
pip install protobuf

pip install lm-eval
pip install ml_collections
```

## Example

The experiment includes three steps.

1) Weight quantization of LLaMA2-7b to 3 bit with `QUIK`.
2) Fine-tuning of columns with outliers of the model based on the instruction dataset `allenai/tulu-v2-sft-mixture`
3) Benchmark of the quantized model after fine-tuning using `lm-evaluation-harness`

### Quantization
```bash
cd QUIK
python python QUIK/experiments/llama.py \
  --model <path to directory with LLaMA2 model> \
  --fp_features 128 \
  --a_bits 16 \
  --w_bits 3 \
  --w_clip \
  --dataset wikitext2
```

The quantized model will be saved in 
'./weights/llama7b_{args.w_bits}w_{args.a_bits}a_{args.fp_features}'

### Fine-tuning
```bash
python llm_tune/train_instruct.py --config_path=llm_tune/configs/llama_instruct.py
```

Before running the script, check the following variables in `llama_instruct.py`
`config.model_name_or_path` is a dicrectory with the quantized model. Tokenizer should be placed in the directory.
`config.max_memory` is maximal memory of one GPU which used for the training. 
`config.output_dir` is a directory where the checkpoints will be saved during the training.
`config.outliers['path_to_act_scales']` is a path to column scales which related to an importance of a column in weight matrix.
The scales for LLaMA2-7b are saved in `/QUIK/experiments/act_scales/Llama-2-7b-hf.pt`.

### Benchmark
Apply `lm-evaluation-harness` to benchmark the quantized model after fine-tuning.
https://github.com/EleutherAI/lm-evaluation-harness/tree/main

```bash
lm_eval --model hf \
  --model_args "pretrained=<path to the directory with quantized model like weights/llama7b_3w_16a_128fp>" \
  --tasks winogrande,hellaswag,swag,boolq,xwinograd_en \
  --batch_size 16 \
  --num_fewshot 0 \
  --device cuda
```
