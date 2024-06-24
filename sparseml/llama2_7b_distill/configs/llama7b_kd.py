import ml_collections


def model_configs():
    config = ml_collections.ConfigDict()


    ### DATASET ###
    data = config.data = ml_collections.ConfigDict()
    # data.dataset = 'oasst1'
    # data.dataset_format = None
    # data.dataset_name = "yahma/alpaca-cleaned"
    # data.dataset_name = "ostapeno/tulu_v2_flan_v2_subset"
    # data.dataset_name = "allenai/tulu-v2-sft-mixture"
    data.dataset_name = "ZhMax/oo-1million-gpt-4-sft"
    # data.dataset_name = "VityaVitalich/LIMA"
    data.dataset_config_name = None
    data.source_max_len = 16
    data.target_max_len = 512
    data.dataset_percentage = 100
    data.validation_split_percentage = 5
    data.max_seq_length = 2048
    data.trust_remote_code = True
    data.preprocessing_num_workers = 8
    data.seed = 11

    ### MODEL CHECKPOINT ###
    config.model_type = 'Auto'
    config.model_name_or_path = '/home/llm_compression/Quantization/SparseGPT/output_llama7b_sparseml/stage_sparsity'
    config.model_config_name = None
    config.tokenizer_name = None
    config.token = None
    config.use_fast_tokenizer = True
    config.trust_remote_code = True
    config.max_memory = 79
    config.teacher_name_or_path = '/home/exp_results/instruct/llama_quik4bit3bit_normal_noise_wanda/merged_500'
    
    ## SAVING DIRS ###
    # config.cache_dir = None
    config.output_dir = '/home/exp_results/sparseml/tutorial/distill/sparsegpt_kd_tulu_noise4bit4bit'

    ### TRAINING ###
    config.run_name = 'outliers_4bit'
    config.resume_from_checkpoint = None
    # config.num_train_epochs = 1
    config.max_steps = 1000
    config.learning_rate = 1e-4
    config.weight_decay = 0.0
    config.lr_scheduler_type = 'linear'
    config.warmup_ratio =  0.03
    config.seed = 11
    config.per_device_train_batch_size = 2
    config.per_device_eval_batch_size = 2
    config.gradient_accumulation_steps = 16
    config.gradient_checkpointing = False
    config.report_to = None

    config.recipe = "/home/LLM_Compression/sparseml/llama2_7b_distill/configs/distill.yaml"
    ### eval ###
    # config.evaluation_strategy = 'steps'
    # config.eval_steps = 100
    # config.evaluation_strategy = 'no'
    # config.eval_steps = None
    ### save ###
    config.save_strategy = 'steps'
    config.save_steps = 125

    ### SOFTMAX CLIP ###
    config.use_clip_softmax = False
    config.clip_softmax_eta = 1
    config.clip_softmax_gamma = -12/512

    ### LORA ###
    config.use_lora = False
    config.lora_rank = 128
    config.lora_alpha = 128
    config.lora_dropout = 0.1
    config.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    config.quant_noise_config = {"quant_bit": 4, "quant_block_size": 128, "outliers_fraction": 0.03}

    ### Outliers
    config.outliers = {
        'path_to_act_scales': '/home/llm_compression/Quantization/Weight_scales/wanda_scales/archive/llama7b_wanda_scales_w4_aMax_ptb.pt',
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
        'path_to_quant_params': '/home/llm_compression/Quantization/Quik/weights/llama7b_4w_16a_weight_scale_wanda_max_ptb/quant_params.pt',
        'learnable_scale': False
    }

    ### SymQuant
    config.SymQuant = {
        'is_quant_weight': False,
        'block_size': 128,
        'learnable_scale': False,
        'layer_bits': {'q': 3, 'k': 3, 'v': 3, 'o': 3, 'down': 3, 'gate': 3, 'up': 3}
    }

    ### NoiseQuant
    config.NoiseQuant = {
        'add_quant_noise': False,
        'predict': False,
        'block_size': 128,
        'compute_scale': False,
        'layer_bits': {'q': 3, 'k': 3, 'v': 3, 'o': 3, 'down': 3, 'gate': 3, 'up': 3}
    }
    ### BitNoiseQuant
    config.BitNoiseQuant = {
        'add_quant_noise': False,
        'predict': False,
        'compute_scale': False,
        'noise_type': 'normal',
        'learnable_scale': False,
        'layer_bits': {'q': 3, 'k': 3, 'v': 3, 'o': 3, 'down': 3, 'gate': 3, 'up': 3}
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
    