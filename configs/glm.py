import ml_collections


def model_configs():
    config = ml_collections.ConfigDict()


    ### DATASET ###
    data = config.data = ml_collections.ConfigDict()
    data.dataset_name = "allenai/tulu-v2-sft-mixture"
    data.dataset_config_name =None #'wikitext-2-raw-v1'
    data.valid_split = 5
    data.block_size = 2048
    data.dataset_percentage = 100
    data.instruct = True

    ### MODEL CHECKPOINT ###
    config.glm = True
    config.version = 4
    config.model_type = 'Auto'
    config.model_name_or_path = "/home/cache/glm4_3bit/"
    config.model_config_name = None
    config.tokenizer_name = None
    config.token = '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'

    ### SAVING DIRS ###
    config.cache_dir = '/home/cache/'
    config.output_dir = '/home/LLM_Compression/ckpts/lora/glm4_3bit_sasut/'
    
    ### TRAINING ###
    config.learning_rate = 1e-4
    config.warmup_steps = 30
    config.weight_decay = 0
    config.seed = 11
    config.num_train_epochs = 1
    config.max_steps = 1000 # set to -1 when not used
    config.per_device_train_batch_size = 2
    config.per_device_eval_batch_size = 2
    config.gradient_accumulation_steps = 16
    config.gradient_checkpointing = False
    config.report_to = 'wandb'
    config.run_name = 'glm4_3bit'
    ### eval ###
    config.evaluation_strategy = 'steps'
    config.eval_steps = 250
    ### save ###
    config.save_strategy = 'steps'
    config.save_steps = 250


    ### SOFTMAX CLIP ###
    config.use_clip_softmax = False
    config.clip_softmax_eta = 1
    config.clip_softmax_gamma = -12/512

    ### LORA ###
    config.use_lora = False
    config.lora_rank = 64
    config.lora_alpha = 16
    config.lora_dropout = 0.1
    config.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    config.dora = False

    ### OUTLIER TUNING = SASUT ###
    config.use_sasut = True
    config.sasut_path_to_act_scales = '/home/LLM_Compression/QUIK/experiments/act_scales/glm4_9b.pt'
    config.sasut_outlier_num = 128
    config.sasut_target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    config.sasut_noise_type = 'normal'
    config.sasut_compute_quant_scale = True
    config.sasut_add_noise = False


    ### NORM TWEEKING ###
    config.norm_tweek = False


    ### ZERO OUTLIERS ###
    config.zero_outliers = False
    config.outlier_fraction = 0.05

    ### STE ###
    ste = config.ste = ml_collections.ConfigDict()
    ste.enable = False
    ste.path_to_act_scales = '/home/LLM_Compression/QUIK/experiments/act_scales/llama3_8b_obs_w2_ptb_max.pt'
    ste.fp_features_num = 128
    ste.layer_bits = {'q': 2, 'k': 2, 'v': 2, 'o': 2, 'down': 2, 'gate': 2, 'up': 2}
    ste.block_size = 64
    ste.learnable_scales = True
    ste.quik_scales_path = '/home/cache/Llama8b_2bit_128fp_owq_max_2/quantazed_model.pt' # either put None

    ### Distillation ###
    config.distillation = False
    config.temperature = 0.8
    config.lambda_param = 0.4
    return config

