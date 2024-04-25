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
    config.model_type = 'Auto'
    config.model_name_or_path = '/home/cache/Llama13b_4bit_128fp_owq_max_4' # "meta-llama/Llama-2-7b-hf"
    config.model_config_name = None
    config.tokenizer_name = None
    config.token = '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'

    ### SAVING DIRS ###
    config.cache_dir = '/home/cache/'
    config.output_dir = '/home/LLM_Compression/ckpts/lora/Llama13b_tulu_quik_4bit_lora_lowlr/'
    
    ### TRAINING ###
    config.learning_rate = 1e-5
    config.weight_decay = 0
    config.seed = 11
    config.num_train_epochs = 1
    config.max_steps = 1000 # set to -1 when not used
    config.per_device_train_batch_size = 1
    config.per_device_eval_batch_size = 2
    config.gradient_accumulation_steps = 16
    config.gradient_checkpointing = False
    config.report_to = 'wandb'
    config.run_name = 'Llama13b_tulu_quik_4bit_lora_lowlr'
    ### eval ###
    config.evaluation_strategy = 'no'
    config.eval_steps = 125
    ### save ###
    config.save_strategy = 'steps'
    config.save_steps = 125


    ### SOFTMAX CLIP ###
    config.use_clip_softmax = False
    config.clip_softmax_eta = 1
    config.clip_softmax_gamma = -12/512

    ### LORA ###
    config.use_lora = True
    config.lora_rank = 64
    config.lora_alpha = 16
    config.lora_dropout = 0.1
    config.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    config.dora = False

    ### NORM TWEEKING ###
    config.norm_tweek = False


    ### ZERO OUTLIERS ###
    config.zero_outliers = False
    config.outlier_fraction = 0.05

    ### STE ###
    ste = config.ste = ml_collections.ConfigDict()
    ste.enable = False
    ste.path_to_act_scales = '/home/LLM_Compression/QUIK/experiments/act_scales/llama13b_owq_w3_ptb_max.pt'
    ste.fp_features_num = 128
    ste.layer_bits = {'q': 3, 'k': 3, 'v': 3, 'o': 3, 'down': 3, 'gate': 3, 'up': 3}
    ste.block_size = 64
    ste.learnable_scales = True
    ste.quik_scales_path = '/home/cache/Llama13b_3bit_128fp_owq_max_3/quantazed_model.pt' # either put None

    ### Distillation ###
    config.distillation = False
    config.temperature = 0.8
    config.lambda_param = 0.4
    return config

