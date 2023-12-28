# LLM_Compression


Evaluation FrameWork = https://github.com/EleutherAI/lm-evaluation-harness/

### DATASETS

1. WinoGrande. https://huggingface.co/datasets/winogrande
2. HellaSwag. https://huggingface.co/datasets/Rowan/hellaswag
3. SWAG. https://huggingface.co/datasets/swag
4. XWinograd En. https://huggingface.co/datasets/Muennighoff/xwinograd
5. BoolQ. Is inside SuperGLUE. https://huggingface.co/datasets/boolq?row=1

### Metrics

| Model         | WinoGrande    | Swag         | HellaSwag    | Xwinograd (en)   | BoolQ        |
| ------------- | ------------- |------------- |------------- |------------- |------------- |
| Llama2-7b     | 0.69          | 0.75 normed    |  0.76 normed | 0.8796  |   0.7777|


### Ideas

1) RL fine tuning with teacher
2) RL with on-policy fine-tuning
3) on-policy find which texts to use for fine-tuning
4) Fine tune with forward passes
5) Online generating texts to fine-tune
6) Project model on lower dimension with procrusted orthogonal problem. To project activations use procrustes on transposed output
