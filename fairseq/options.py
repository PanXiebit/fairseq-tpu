
class Config():
    # load dataset
    data = "output/data-raw"
    source_lang = "en"
    target_lang = "ro"
    left_pad_source = True
    left_pad_target = False
    add_bos = True
    raw_text = True  # data save/load format

    input_shapes = [[256, 16], [128, 32], [64, 64]]
    max_source_positions = 1000
    max_target_positions = 1000
    max_tokens = 4096
    max_sentences = None
    dynamic_length = False
    mask_range = False
    upsample_primary = 1
    workers = 8

    ## model parameters
    share_all_embeddings = True
    encoder_embed_dim = 512
    decoder_embed_dim = 512
    encoder_embed_path = None
    decoder_embed_path = None
    encoder_embed_scale = None  # TODO?
    decoder_embed_scale = None
    dropout = 0.3
    no_enc_token_positional_embeddings = False
    no_dec_token_positional_embeddings = False
    encoder_layers = 6
    decoder_layers = 6
    encoder_attention_heads = 8
    decoder_attention_heads = 8
    attention_dropout = 0.0
    relu_dropout = 0.0
    encoder_normalize_before = False
    decoder_normalize_before = False
    encoder_ffn_embed_dim = 2048
    decoder_ffn_embed_dim = 2048
    sharing_gen_dis = True
    decoder_output_dim = 512

    # criterion
    label_smoothing = 0.1
    gen_weights = 1.0
    dis_weights = 1.0

    # optimizer
    lr = [0.0005]
    lr_scheduler = 'inverse_sqrt'
    adam_betas = '(0.9, 0.999)'
    adam_eps = 1e-06
    warmup_init_lr = 1e-07
    warmup_updates = 10000
    weight_decay = 0.01

    log_steps = 100
    checkpoint_path = "output/my_maskPredict_en_ro"
    num_epochs = 100
    num_cores = 8

    use_gpu = True






    





args = Config()

