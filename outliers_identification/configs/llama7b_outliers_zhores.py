import ml_collections

def model_configs():
    config = ml_collections.ConfigDict()


    ### DATASET ###
    data = config.data = ml_collections.ConfigDict()
    data.dataset_path = "/home/llm_compression/Quantization/Weight_scales/datasets/val.jsonl.zst"
    data.output_path = "/home/llm_compression/Quantization/Weight_scales/obs_scales/llama7b_obs_scales_2bit.pt"
    data.max_seq_length = 512
    data.num_samples = 512
    data.trust_remote_code = True
    data.preprocessing_num_workers = 8

    ### MODEL CHECKPOINT ###
    config.model_type = 'Auto'
    config.model_name_or_path = '/home/llm_compression/LLaMA/Llama-2-7b-hf'
    config.token = None
    config.use_fast_tokenizer = True
    config.trust_remote_code = True

    ### Quantization ###
    config.bit = 2

    return config