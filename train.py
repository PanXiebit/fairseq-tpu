from fairseq.options import args
from fairseq.datasets import TranslationSelfDataset, get_batch_iterator
from fairseq import utils
from fairseq.models import Transformer_nonautoregressive_gan
from fairseq.criterions import LabelSmoothedLengthGan_CrossEntropyCriterion
# load dataset
translation_self = TranslationSelfDataset.load_dictionary(args)
valid_dataset = translation_self.load_dataset("valid")
# train_dataset = translation_self.load_dataset("train")


valid_dataloader = get_batch_iterator(
    valid_dataset,
    input_shapes=args.input_shapes,
    max_tokens=args.max_tokens,
    max_positions=utils.resolve_max_positions(
                translation_self.max_positions(),
                (args.max_source_positions, args.max_target_positions)))
# train_dataloader = get_batch_iterator(
#     train_dataset,
#     input_shapes=args.input_shapes,
#     max_tokens=args.max_tokens,
#     max_positions=utils.resolve_max_positions(
#                 translation_self.max_positions(),
#                 (args.max_source_positions, args.max_target_positions)))
model = Transformer_nonautoregressive_gan.build_model(
    args, translation_self.src_dict, translation_self.tgt_dict)
criterion = LabelSmoothedLengthGan_CrossEntropyCriterion(args, translation_self.tgt_dict)

print(model, criterion)

for i, sample in enumerate(valid_dataloader):
    # if i > 3:
    #     break
    loss, sample_size, logging_output = criterion(model, sample)
    logging= criterion.aggregate_logging_outputs([logging_output])
    print(logging["loss"], logging["nll_loss"], logging["length_loss"], logging["dis_loss"])