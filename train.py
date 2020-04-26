








from fairseq.options import args
from fairseq.datasets import TranslationSelfDataset, get_batch_iterator

# load dataset

translation_self = TranslationSelfDataset.load_dictionary(args)
train_dataset = translation_self.load_dataset("train")
valid_dataset = translation_self.load_dataset("valid")
train_iter = get_batch_iterator(train_dataset)

