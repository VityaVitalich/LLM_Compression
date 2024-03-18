import torch
from transformers import AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer, AutoConfig, PretrainedConfig
from peft import PeftModel 


if __name__ == '__main__':
    checkpoint = 'meta-llama/Llama-2-7b-hf'
    token = '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' 
    
    base_model = LlamaForCausalLM.from_pretrained(
            checkpoint,
            device_map='cpu',
            use_auth_token=token)

    ls = ['349', '523', '698', '870']
    for ckpt in ls:
        peft_model_id = f"/home/data/compression/clip_sm_cache/fine_tuning/lora/checkpoint-{ckpt}" 


        model = PeftModel.from_pretrained(base_model, peft_model_id) 
        merged_model = model.merge_and_unload()

        merged_model.save_pretrained(f'/home/data/compression/clip_sm_cache/fine_tuning/lora/ckpt{ckpt}_sm_gamma-2e-2', from_pt=True)

        del model
        del merged_model