import ml_collections


def model_configs():
    config = ml_collections.ConfigDict()


    ### DATASET ###
    data = config.data = ml_collections.ConfigDict()
    # data.dataset_name = 'togethercomputer/RedPajama-Data-1T-Sample'
    # data.dataset_config_name = None

    data.dataset_name = 'wikitext'
    data.dataset_config_name = 'wikitext-2-raw-v1'
    data.dataset_percentage = 10
    data.validation_split_percentage = 5
    data.block_size = 128
    data.trust_remote_code = True
    data.preprocessing_num_workers = 8

    ### MODEL CHECKPOINT ###
    config.model_type = 'Auto'
    # config.model_name_or_path = '/home/llm_compression/LLaMA/Llama-2-7b-hf'
    config.model_name_or_path = '/home/projects/LLaMA/huggingface/Llama-2-7b-hf'
    config.model_config_name = None
    config.tokenizer_name = None
    config.token = 'YOUR TOKEN'
    config.use_fast_tokenizer = True
    config.trust_remote_code = True

    ### SAVING DIRS ###
    config.cache_dir = None
    config.output_dir = './exp_results/red_pajama/llama_with_noise2bit'
    
    ### TRAINING ###
    config.run_name = 'lora_noise'
    config.resume_from_checkpoint = None
    config.num_train_epochs = None
    config.max_steps= 2000
    config.learning_rate = 8e-4
    config.weight_decay = 1e-3
    config.lr_scheduler_type = 'cosine'
    config.warmup_ratio =  0.03
    config.seed = 11
    config.per_device_train_batch_size = 8
    config.per_device_eval_batch_size = 8
    config.gradient_accumulation_steps = 16
    config.gradient_checkpointing = False
    config.report_to = 'wandb'
    ### eval ###
    config.evaluation_strategy = 'steps'
    config.eval_steps = 100
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
    config.quant_noise_config = {"quant_bit": 2, "quant_block_size": 128, "outliers_fraction": 0.03}

    ### NORM TWEEKING ###
    config.norm_tweek = True


    ### ZERO OUTLIERS ###
    config.zero_outliers = False
    config.outlier_fraction = 0.05
    
    return config