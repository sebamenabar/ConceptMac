import os

import glob
import errno
import pickle
from copy import deepcopy

import numpy as np

import torch
import torch.nn as nn
from torch.nn import init
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.nn import functional as F

from config import cfg


def save_model(model, optim, iter, model_dir, max_to_keep=None, model_name=""):
    checkpoint = {
        'iter': iter,
        'model': model.state_dict(),
        'optim': optim.state_dict() if optim is not None else None}
    if model_name == "":
        torch.save(checkpoint, "{}/checkpoint_{:06}.pth".format(model_dir, iter))
    else:
        torch.save(checkpoint, "{}/{}_checkpoint_{:06}.pth".format(model_dir, model_name, iter))

    if max_to_keep is not None and max_to_keep > 0:
        checkpoint_list = sorted([ckpt for ckpt in glob.glob(model_dir + "/" + '*.pth')])
        while len(checkpoint_list) > max_to_keep:
            os.remove(checkpoint_list[0])
            checkpoint_list = checkpoint_list[1:]


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def init_modules(modules, w_init='kaiming_uniform'):
    if w_init == "normal":
        _init = init.normal_
    elif w_init == "xavier_normal":
        _init = init.xavier_normal_
    elif w_init == "xavier_uniform":
        _init = init.xavier_uniform_
    elif w_init == "kaiming_normal":
        _init = init.kaiming_normal_
    elif w_init == "kaiming_uniform":
        _init = init.kaiming_uniform_
    elif w_init == "orthogonal":
        _init = init.orthogonal_
    else:
        raise NotImplementedError
    for m in modules:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            _init(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        if isinstance(m, (nn.LSTM, nn.GRU)):
            for name, param in m.named_parameters():
                if 'bias' in name:
                    nn.init.zeros_(param)
                elif 'weight' in name:
                    _init(param)


def load_vocab(cfg):
    def invert_dict(d):
        return {v: k for k, v in d.items()}

    with open(os.path.join(cfg.DATASET.DATA_DIR, 'dic.pkl'), 'rb') as f:
        dictionaries = pickle.load(f)
    vocab = {}
    vocab['question_token_to_idx'] = dictionaries["word_dic"]
    vocab['answer_token_to_idx'] = dictionaries["answer_dic"]
    vocab['question_token_to_idx']['pad'] = 0
    vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
    vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])

    return vocab


def generateVarDpMask(shape, keepProb, device=None):
    randomTensor = torch.tensor(keepProb).expand(shape)
    randomTensor = randomTensor + nn.init.uniform_(torch.FloatTensor(shape[0], shape[1]))
    binaryTensor = torch.floor(randomTensor)
    mask = torch.FloatTensor(binaryTensor)
    mask = mask.to(device)
    return mask


def applyVarDpMask(inp, mask, keepProb):
    ret = (torch.div(inp, torch.tensor(keepProb, device=inp.device))) * mask
    return ret

def cfg_to_exp_name(cfg):
    bsz = cfg.TRAIN.BATCH_SIZE
    lr = cfg.TRAIN.LEARNING_RATE
    module_dim = cfg.model.common.module_dim
    max_step = cfg.model.max_step
    sss = 'sss' if cfg.model.separate_syntax_semantics else ''
    if len(sss) and cfg.model.input_unit.separate_syntax_semantics_embeddings:
        sss += 'e'
    control_feed_prev = cfg.model.control_unit.control_feed_prev
    
    if cfg.model.write_unit.rtom:
        write = 'rtom'
    else:
        write = ''
        if cfg.model.write_unit.self_attn:
            write += 'sa'
        if cfg.model.write_unit.gate:
            write += 'g'
            if cfg.model.write_unit.gate_shared:
                write += 's'
            else:
                write += 'u'
    
    exp_name = f'{max_step}_{module_dim}'
    if sss:
        exp_name += f'_{sss}'
    if control_feed_prev:
        exp_name += f'_cfp'
    exp_name += f'_{write}'

    exp_name += f'_bsz{bsz}_lr{lr}'

    return exp_name
            
        