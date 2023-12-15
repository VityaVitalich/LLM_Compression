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
| Llama2-7b     | 0.69          | 0.75 normed    |  0.76 normed | 0.8796  |   |
