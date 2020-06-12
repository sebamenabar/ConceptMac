import os

import glob
import errno
import pickle
import functools
from copy import deepcopy
from collections import OrderedDict
from itertools import chain, starmap

import numpy as np

import torch
import torch.nn as nn
from torch.nn import init
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.nn import functional as F

from config import cfg


def idxs_to_question(idxs, mapping):
    return [mapping[idx] for idx in idxs]


def save_model(model, optim, iter, model_dir, max_to_keep=None, model_name=""):
    checkpoint = {
        "iter": iter,
        "model": model.state_dict(),
        "optim": optim.state_dict() if optim is not None else None,
    }
    if model_name == "":
        torch.save(checkpoint, "{}/checkpoint_{:06}.pth".format(model_dir, iter))
    else:
        torch.save(
            checkpoint, "{}/{}_checkpoint_{:06}.pth".format(model_dir, model_name, iter)
        )

    if max_to_keep is not None and max_to_keep > 0:
        checkpoint_list = sorted(
            [ckpt for ckpt in glob.glob(model_dir + "/" + "*.pth")]
        )
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


def init_modules(modules, w_init="kaiming_uniform"):
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
                if "bias" in name:
                    nn.init.zeros_(param)
                elif "weight" in name:
                    _init(param)


def load_vocab(base_dir, data_folder="data"):
    def invert_dict(d):
        return {v: k for k, v in d.items()}

    with open(os.path.join(base_dir, data_folder, "dic.pkl"), "rb") as f:
        dictionaries = pickle.load(f)
    vocab = {}
    vocab["question_token_to_idx"] = dictionaries["word_dic"]
    vocab["answer_token_to_idx"] = dictionaries["answer_dic"]
    vocab["question_token_to_idx"]["<PAD>"] = 0
    vocab["question_idx_to_token"] = invert_dict(vocab["question_token_to_idx"])
    vocab["answer_idx_to_token"] = invert_dict(vocab["answer_token_to_idx"])

    return vocab


def generateVarDpMask(shape, keepProb, device=None):
    randomTensor = torch.tensor(keepProb).expand(shape)
    randomTensor = randomTensor + nn.init.uniform_(
        torch.FloatTensor(shape[0], shape[1])
    )
    binaryTensor = torch.floor(randomTensor)
    mask = torch.FloatTensor(binaryTensor)
    mask = mask.to(device)
    return mask


def applyVarDpMask(inp, mask, keepProb):
    ret = (torch.div(inp, torch.tensor(keepProb, device=inp.device))) * mask
    return ret


def cfg_to_exp_name(cfg):
    # Common config
    bsz = cfg.TRAIN.BATCH_SIZE
    lr = cfg.TRAIN.LEARNING_RATE
    module_dim = cfg.model.common.module_dim
    max_step = cfg.model.max_step
    sss = "sss" if cfg.model.separate_syntax_semantics else ""
    if len(sss) and cfg.model.input_unit.separate_syntax_semantics_embeddings:
        sss += "e"
    use_feats = "objs" if cfg.model.use_feats == "objects" else ""
    # Control config
    control_feed_prev = cfg.model.control_unit.control_feed_prev
    # Read config
    read_gate = cfg.model.read_unit.gate
    num_lobs = cfg.model.read_unit.num_lobs or cfg.model.num_gt_lobs
    # Write config
    if cfg.model.write_unit.rtom:
        write = "rtom"
    else:
        write = ""
        if cfg.model.write_unit.self_attn:
            write += "sa"
        if cfg.model.write_unit.gate:
            write += "g"
            if cfg.model.write_unit.gate_shared:
                write += "s"
            else:
                write += "u"
    if len(write) == 0:
        write = "std"

    exp_name = f"{max_step}"
    if sss:
        exp_name += f"_{sss}"
    if control_feed_prev:
        exp_name += f"_cfp"
    if read_gate or use_feats == "objs":
        exp_name += f"_gatelobs{cfg.model.read_unit.num_lobs}"
    if cfg.model.num_gt_lobs > 0:
        exp_name += f"_gt{cfg.model.num_gt_lobs}"
    exp_name += f"_{write}"
    if use_feats:
        exp_name += f"_{use_feats}"

    exp_name += f"_{module_dim}"
    # exp_name += f'_{module_dim}_bsz{bsz}_lr{lr}'

    return exp_name


def flatten_json_iterative_solution(dictionary):
    """Flatten a nested json file"""

    def unpack(parent_key, parent_value):
        """Unpack one level of nesting in json file"""
        # Unpack one level only!!!

        if isinstance(parent_value, dict):
            for key, value in parent_value.items():
                temp1 = parent_key + "." + key
                yield temp1, value
        elif isinstance(parent_value, list):
            i = 0
            for value in parent_value:
                temp2 = parent_key + "." + str(i)
                i += 1
                yield temp2, value
        else:
            yield parent_key, parent_value

    # Keep iterating until the termination condition is satisfied
    while True:
        # Keep unpacking the json file until all values are atomic elements (not dictionary or list)
        dictionary = dict(chain.from_iterable(starmap(unpack, dictionary.items())))
        # Terminate condition: not any value in the json file is dictionary or list
        if not any(
            isinstance(value, dict) for value in dictionary.values()
        ) and not any(isinstance(value, list) for value in dictionary.values()):
            break

    return dictionary


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


class IntermediateLayerGetter(nn.Module):
    def __init__(self, model, return_layers, keep_output=True):
        super().__init__()
        self._model = model
        self.return_layers = return_layers
        self.keep_output = keep_output

    def forward(self, *args, **kwargs):
        ret = OrderedDict()
        handles = []
        for name, new_name in self.return_layers.items():
            layer = rgetattr(self._model, name)

            def hook(module, input, output, new_name=new_name):
                if new_name in ret:
                    if type(ret[new_name]) is list:
                        ret[new_name].append(output)
                    else:
                        ret[new_name] = [ret[new_name], output]
                else:
                    ret[new_name] = output

            try:
                h = layer.register_forward_hook(hook)
            except AttributeError as e:
                raise AttributeError(f"Module {name} not found in model")
            handles.append(h)

        if self.keep_output:
            output = self._model(*args, **kwargs)
        else:
            self._model(*args, **kwargs)
            output = None

        for h in handles:
            h.remove()

        return ret, output
