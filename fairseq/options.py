
class Config():
    # load dataset
    data = "output/data-bin"
    source_lang = "en"
    target_lang = "ro"
    left_pad_source = True
    left_pad_target = False
    add_bos = True
    raw_text = False  # data save/load format

    max_source_positions = 1000
    max_target_positions = 1000
    max_tokens = 4096
    max_sentences = None
    dynamic_length = False
    mask_range = False
    upsample_primary = 1
    workers = 8


args = Config()

