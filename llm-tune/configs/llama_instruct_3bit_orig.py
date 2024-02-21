import ml_collections


def model_configs():
    config = ml_collections.ConfigDict()


    ### DATASET ###
    data = config.data = ml_collections.ConfigDict()
    # data.dataset = 'oasst1'
    # data.dataset_format = None
    # data.dataset_name = "yahma/alpaca-cleaned"
    data.dataset_name = "ostapeno/tulu_v2_flan_v2_subset"
    data.dataset_config_name = None
    data.source_max_len = 16
    data.target_max_len = 512
    data.dataset_percentage = 100
    data.validation_split_percentage = 5
    data.max_seq_length = 2048
    data.trust_remote_code = True
    data.preprocessing_num_workers = 8


    ### MODEL CHECKPOINT ###
    config.model_type = 'Auto'
    config.model_name_or_path = '/home/llm_compression/Quantization/Quik/weights/llama7b_3w_16a_128fp'
    config.model_config_name = None
    config.tokenizer_name = None
    config.token = None
    config.use_fast_tokenizer = True
    config.trust_remote_code = True
    config.max_memory = 79

    ### SAVING DIRS ###
    # config.cache_dir = None
    config.output_dir = './exp_results/instruct/quik3bit_flan'

    ### TRAINING ###
    config.run_name = 'linearquant_3bit'
    config.resume_from_checkpoint = None
    # config.num_train_epochs = None
    config.max_steps = 4000
    config.learning_rate = 1e-4
    config.weight_decay = 0.1
    config.lr_scheduler_type = 'linear'
    config.warmup_ratio =  0.03
    config.seed = 11
    config.per_device_train_batch_size = 4
    config.per_device_eval_batch_size = 4
    config.gradient_accumulation_steps = 8
    config.gradient_checkpointing = False
    config.report_to = None
    ### eval ###
    # config.evaluation_strategy = 'steps'
    # config.eval_steps = 100
    # config.evaluation_strategy = 'no'
    # config.eval_steps = None
    ### save ###
    config.save_strategy = 'steps'
    config.save_steps = 200


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
    config.quant_noise_config = {"quant_bit": 3, "quant_block_size": 128, "outliers_fraction": 0.03}

    ### LinearQuantNoise
    config.LinearQuantNoise = {
        'replace_Linear': True,
        'path_to_act_scales': "/home/LLM_Compression/QUIK/experiments/act_scales/Llama-2-7b-hf.pt",
        'fp_features_num': 128,
        'quant_target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        'quant_bit': 3,
        'block_size': 128, 
        'training_mode': "train_fp_weight", 
        'add_quant_noise': False
    }


    ### NORM TWEEKING ###
    config.norm_tweek = True

    ###LM HEAD ###
    config.train_lm_head = False

    ### ZERO OUTLIERS ###
    config.zero_outliers = False
    config.outlier_fraction = 0.05
    
    return config