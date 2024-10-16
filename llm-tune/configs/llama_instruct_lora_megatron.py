import ml_collections


def model_configs():
    config = ml_collections.ConfigDict()


    ### DATASET ###
    data = config.data = ml_collections.ConfigDict()
    # data.dataset = 'oasst1'
    # data.dataset_format = None
    # data.dataset_name = "yahma/alpaca-cleaned"
    # data.dataset_name = "ostapeno/tulu_v2_flan_v2_subset"
    data.dataset_name = "allenai/tulu-v2-sft-mixture"
    data.dataset_config_name = None
    data.source_max_len = 16
    data.target_max_len = 512
    data.dataset_percentage = 10
    data.validation_split_percentage = 5
    data.max_seq_length = 128 #2048
    data.trust_remote_code = True
    data.preprocessing_num_workers = 8
    data.seed = 11


    ### MODEL CHECKPOINT ###
    config.model_type = 'Auto'
    config.model_name_or_path = '/home/LLaMA/huggingface/Llama-2-7b-hf'
    # config.model_name_or_path = '/home/Quantization/weights_study/weights/llama7b_mixed_weight'
    # config.model_name_or_path = '/home/Quantization/weights_study/weights/llama7b_3bit_loaded'
    # config.model_name_or_path = '/home/Quantization/weights_study/weights/llama_quik3bit_transformers_sft_mixture_only_fp_quant_split_weight/checkpoint-200'
    # config.model_name_or_path = '/home/LLM_compression/QUIK/weights/llama7b_3bit_128fp_quant_scales'
    # config.model_name_or_path = '/home/exp_results/output/instruct/llama7b_test_outliers/checkpoint-2'
    # config.model_name_or_path = '/home/Quantization/weights_study/weights/checkpoint-600'
    config.model_config_name = None
    config.tokenizer_name = None
    config.token = None
    config.use_fast_tokenizer = True
    config.trust_remote_code = True
    config.max_memory = 18

    ## SAVING DIRS ###
    # config.cache_dir = None
    config.output_dir = '/home/exp_results/output/instruct/llama7b_test_outliers'

    ### TRAINING ###
    config.run_name = 'linearquant_3bit'
    config.resume_from_checkpoint = None
    # config.num_train_epochs = None
    config.max_steps = 2
    config.learning_rate = 1e-2
    config.weight_decay = 0.0
    config.lr_scheduler_type = 'linear'
    config.warmup_ratio =  0.03
    config.seed = 11
    config.per_device_train_batch_size = 1
    config.per_device_eval_batch_size = 1
    config.gradient_accumulation_steps = 16
    config.gradient_checkpointing = False
    config.report_to = None
    ### eval ###
    # config.evaluation_strategy = 'steps'
    # config.eval_steps = 100
    # config.evaluation_strategy = 'no'
    # config.eval_steps = None
    ### save ###
    config.save_strategy = 'steps'
    config.save_steps = 1

    ### SOFTMAX CLIP ###
    config.use_clip_softmax = False
    config.clip_softmax_eta = 1
    config.clip_softmax_gamma = -12/512

    ### LORA ###
    config.use_lora = True
    config.lora_rank = 128
    config.lora_alpha = 128
    config.lora_dropout = 0.1
    config.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    config.path_to_file_for_lora_init = '/home/exp_results/llama7b_fp64_quik3bit_svd'
    config.quant_noise_config = {"quant_bit": 4, "quant_block_size": 128, "outliers_fraction": 0.03}

    ### Outliers
    config.outliers = {
        'path_to_act_scales': '/home/LLM_compression/QUIK/experiments/act_scales/Llama-2-7b-hf.pt',
        'fp_features_num': 128, 
    }

    ### QuantizedLinear
    config.QuantizedLinear = {
        'replace': False,
        'training_mode': 'train_outlier' #train_full, train_outlier, train_quant
    }

    ### Load Quantized Weight After Quik
    config.loading_quik_quant_weight = {
        'load_weight': False,
        'path_to_quant_params': '/home/LLM_compression/QUIK/weights/llama7b_3bit_128fp_quant_scales/quant_params.pt',
        'learnable_scale': False
    }

    ### SymQuant
    config.SymQuant = {
        'is_quant_weight': False,
        'learnable_scale': False,
        'layer_bits': {'q': 3, 'k': 3, 'v': 3, 'o': 3, 'down': 3, 'gate': 3, 'up': 3}
    }

    ### NoiseQuant
    config.NoiseQuant = {
        'add_quant_noise': False,
        'predict': False,
        'block_size': 128,
        'compute_scale': False,
        'layer_bits': {'q': 2, 'k': 2, 'v': 2, 'o': 2, 'down': 2, 'gate': 2, 'up': 2}
    }

    ### BitNoiseQuant
    config.BitNoiseQuant = {
        'add_quant_noise': False,
        'predict': False,
        'compute_scale': False,
        'learnable_scale': False,
        'noise_type': 'normal',
        'layer_bits': {'q': 2, 'k': 2, 'v': 2, 'o': 2, 'down': 2, 'gate': 2, 'up': 2}
    }

    ### NORM TWEEKING ###
    config.norm_tweek = False

    ###LM HEAD ###
    config.train_lm_head = False

    ### ZERO OUTLIERS ###
    config.zero_outliers = False
    config.outlier_fraction = 0.05

    ### STE ###
    # ste = config.ste = ml_collections.ConfigDict()
    # ste.enable = True
    # ste.path_to_act_scales = '/home/LLM_Compression/QUIK/experiments/act_scales/Llama-2-7b-hf.pt'
    # ste.fp_features_num = 128
    # ste.layer_bits = {'q': 2, 'k': 4, 'v': 4, 'o': 4, 'down': 4, 'gate': 4, 'up': 4}
    # ste.block_size = 64
    
    return config