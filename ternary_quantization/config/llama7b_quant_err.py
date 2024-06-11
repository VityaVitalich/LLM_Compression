import ml_collections

def model_configs():
    config = ml_collections.ConfigDict()


    ### DATASET ###
    data = config.data = ml_collections.ConfigDict()
    data.dataset_path = "/home/llm_compression/Quantization/Weight_scales/datasets/val.jsonl.zst"
    data.output_path = "/home/exp_results/adapters/init_adapters_64rank_for_lora7b_2w"
    data.max_seq_length = 512
    data.num_samples = 512
    data.trust_remote_code = True
    data.preprocessing_num_workers = 8

    ### MODEL CHECKPOINT ###
    config.model_type = 'Auto'
    config.model_orig_path = '/home/llm_compression/LLaMA/Llama-2-7b-hf'
    config.quant_params_path = '/home/llm_compression/Quantization/Quik/weights/llama7b_w2_a16_obs_owq_max/quant_params.pt'
    config.token = None
    config.use_fast_tokenizer = True
    config.trust_remote_code = True

    ### Estimator ###
    config.estimator = {
        'bit': 3,
        'path_to_cols_metric': '/home/LLM_compression/QUIK/experiments/act_scales/Llama-2-7b-hf.pt',
        'fp_features': 128,
        'adapter_rank': 64
    }

    return config