import ml_collections


def model_configs():
    config = ml_collections.ConfigDict()


    ### DATASET ###
    data = config.data = ml_collections.ConfigDict()
    data.dataset_name = 'wikitext'
    data.dataset_config_name = 'wikitext-2-raw-v1'
    data.valid_split = 5
    data.block_size = 3
    data.dataset_percentage = 5

    ### MODEL CHECKPOINT ###
    config.model_type = 'Auto'
    config.model_name_or_path = 'meta-llama/Llama-2-7b-hf'
    config.model_config_name = None
    config.tokenizer_name = None
    config.token = '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'

    ### SAVING DIRS ###
    config.cache_dir = '/home/data/taxonomy/hf_cache/'
    config.output_dir = '/home/LLM_Compression/logs/fine_tuning/full/test/'
    
    ### TRAINING ###
    config.learning_rate = 3e-5
    config.weight_decay = 1e-3
    config.seed = 57
    config.num_train_epochs = 1
    config.per_device_train_batch_size = 1
    config.per_device_eval_batch_size = 1
    config.gradient_accumulation_steps = 2
    config.gradient_checkpointing = False
    config.report_to = 'wandb'
    config.run_name = 'test_STE'
    ### eval ###
    config.evaluation_strategy = 'steps'
    config.eval_steps = 100
    ### save ###
    config.save_strategy = 'steps'
    config.save_steps = 100


    ### SOFTMAX CLIP ###
    config.use_clip_softmax = False
    config.clip_softmax_eta = 1
    config.clip_softmax_gamma = -12/512

    ### LORA ###
    config.use_lora = False
    config.lora_rank = 32
    config.lora_alpha = 16
    config.lora_dropout = 0.1
    config.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]

    ### NORM TWEEKING ###
    config.norm_tweek = False


    ### ZERO OUTLIERS ###
    config.zero_outliers = False
    config.outlier_fraction = 0.05

    ### STE ###
    ste = config.ste = ml_collections.ConfigDict()
    ste.enable = True
    ste.path_to_act_scales = '/home/LLM_Compression/QUIK/experiments/act_scales/Llama-2-7b-hf.pt'
    ste.fp_features_num = 128
    ste.layer_bits = {'q': 2, 'k': 4, 'v': 4, 'o': 4, 'down': 4, 'gate': 4, 'up': 4}
    ste.block_size = 64
    return config
