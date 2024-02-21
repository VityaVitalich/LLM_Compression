# QUIK
This directory contains the code for fake quantization of LLaMA with QUIK, a method for quantizing the majority of the weights post-training.

QUIK is described in the following paper: 
https://arxiv.org/abs/2310.09259

The original repository:
clone https://github.com/IST-DASLab/QUIK.git

## Install

No installation required


## Example

### LLama example
```bash
cd experiments
python python llama.py --model <path to directory with LLaMA model> --fp_features 128 --a_bits 16 --w_bits 3 --w_clip --dataset wikitext2 
```

The quantized model will be saved in 
```
"./weights/llama7b_{args.w_bits}w_{args.a_bits}a_{args.fp_features}"
```

### Citation 

The full paper is available on arxiv. The full citation is

```
@article{QUIK,
  title={QUIK: Towards End-to-end 4-Bit Inference on Generative Large Language Models},
  author={Ashkboos, Saleh and Markov, Ilia and Frantar, Elias and Zhong, Tingxuan and Wang, Xincheng and Ren, Jie and Hoefler, Torsten and Alistarh, Dan},
  journal={arXiv preprint arXiv:2310.09259},
  year={2023}
}
```
