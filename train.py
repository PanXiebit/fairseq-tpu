
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils
import torch_xla.distributed.data_parallel as dp

import warnings
warnings.filterwarnings("ignore")

import argparse
import io
import os
import psutil
import torch
import pandas as pd
import numpy as np
import joblib
import logging
import sys
import torch.nn as nn
import torch.optim as optim
import gc
import time

from fairseq.options import args
from fairseq.datasets import TranslationSelfDataset, get_batch_iterator
from fairseq import utils
from fairseq.models import Transformer_nonautoregressive_gan
from fairseq.criterions import LabelSmoothedLengthGan_CrossEntropyCriterion
from fairseq.optim import FairseqAdam
from fairseq.optim.lr_scheduler  import InverseSquareRootSchedule

args.use_gpu = False

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

_MODEL = None
def create_model(args):
    global _MODEL
    _MODEL = Transformer_nonautoregressive_gan.build_model(
        args, translation_self.src_dict, translation_self.tgt_dict)
    print('PF-MEM: {}'.format(psutil.virtual_memory()))

def _prepare_sample(sample, device):
    if sample is None or len(sample) == 0:
        return None
    if args.use_gpu:
        sample = utils.move_to_cuda(sample)
    else:
        sample = utils.move_to_tpu(sample, device)
    def apply_half(t):
        if t.dtype is torch.float32:
            return t.half()
        return t
    if args.fp16:
        sample = utils.apply_to_sample(apply_half, sample)
    return sample

def train_loop_fn(train_loader, args, model, criterion, optimizer, device, scheduler=None):
    model.train()
    criterion.train()
    for i, sample in enumerate(train_loader):
        sample = _prepare_sample(sample, device)
        print(sample["target"].shape, sample["target"].device)
        optimizer.zero_grad()
        _, _, logging_output = criterion(model, sample)
        logging = criterion.aggregate_logging_outputs([logging_output])
        loss = logging["loss"]
        loss.backward()
        xm.optimizer_step(optimizer, barrier=True)
        if i % args.log_steps == 0:
            xm.master_print('bi={}, loss={:.4f}'.format(i, loss.item()))
            xm.master_print('MEM: {}'.format(psutil.virtual_memory()))
    print('End training: {}'.format(device))


def eval_loop_fn(valid_loader, args, model, criterion, device):
    model.eval()
    criterion.eval()
    valid_logging = []
    for i, sample in enumerate(valid_loader):
        sample = _prepare_sample(sample, device)
        _, _, logging_output = criterion(model, sample)
        valid_logging.append(logging_output)
    val_log = criterion.aggregate_logging_outputs(valid_logging)
    return val_log


import multiprocessing
_LOAD_LOCK = multiprocessing.Lock()
print(_LOAD_LOCK)

def _mp_fn(rank, args):
    print("rank", rank)
    device = xm.xla_device()
    # devices = (
    #   xm.get_xla_supported_devices(
    #       max_devices=args.num_cores) if args.num_cores != 0 else [])
    # with _LOAD_LOCK:
    #     _MODEL.to(device)
    xm.master_print('done loading model')

    criterion = LabelSmoothedLengthGan_CrossEntropyCriterion(args, translation_self.tgt_dict)

    params = list(filter(lambda p: p.requires_grad, _MODEL.parameters()))
    optimizer = FairseqAdam(args, params)
    lr_scheduler = InverseSquareRootSchedule(args, optimizer)

    for epoch in range(args.num_epochs):
        # train_loop_fn(args, _MODEL, criterion, optimizer, device)
        # valid_log = eval_loop_fn(args, _MODEL, criterion, device)
        para_loader = pl.ParallelLoader(valid_dataloader, [device])
        train_loop_fn(para_loader.per_device_loader(device), args, _MODEL, criterion, optimizer, device)
        para_loader = pl.ParallelLoader(valid_dataloader, [device])
        valid_log = eval_loop_fn(para_loader.per_device_loader(device), args, _MODEL, criterion, device)
        xm.master_print('Finished training epoch {}'.format(epoch))

        xm.master_print("Epoch {}, loss {:.4f}, nll_loss {:.4f}, length_loss {:.4f}, dis_loss {:.4f}"
                        .format(epoch, valid_log["loss"], valid_log["nll_loss"],
                                valid_log["length_loss"], valid_log["dis_loss"]))
        lr_scheduler.step(epoch)
        if args.checkpoint_path:
            xm.save(_MODEL.state_dict(), args.checkpoint_path)

if __name__  == "__main__":
    create_model(args)
    xmp.spawn(_mp_fn, args=(args,), nprocs=args.num_cores, start_method='fork')




