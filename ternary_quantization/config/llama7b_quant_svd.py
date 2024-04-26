import ml_collections

def model_configs():
    config = ml_collections.ConfigDict()


    ### DATASET ###
    data = config.data = ml_collections.ConfigDict()
    data.dataset_path = "/home/llm_compression/Quantization/Weight_scales/datasets/val.jsonl.zst"
    data.output_path = "/home/exp_results/llama7b_fp64_quik3bit_svd"
    data.max_seq_length = 512
    data.num_samples = 512
    data.trust_remote_code = True
    data.preprocessing_num_workers = 8

    ### MODEL CHECKPOINT ###
    config.model_type = 'Auto'
    config.model_name_or_path = '/home/LLaMA/huggingface/Llama-2-7b-hf'
    config.token = None
    config.use_fast_tokenizer = True
    config.trust_remote_code = True

    ### Estimator ###
    config.estimator = {
        'bit': 2,
        'path_to_cols_metric': '/home/LLM_compression/QUIK/experiments/act_scales/Llama-2-7b-hf.pt',
        'fp_features': 64,
        'rank_reduction': 16
    }

    return config