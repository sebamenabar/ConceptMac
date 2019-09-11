#!/usr/bin/env python
# coding: utf-8

import os
import sys

import json
import random
import pickle
import argparse
import functools
from collections import Counter

import nltk
import pandas
import numpy as np
from PIL import Image, ImageEnhance

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LinearSegmentedColormap

from tqdm import tqdm


def get_im_features(impath):
    img = PILImage.open(impath).convert('RGB')
    img = transform(img)
    img = img.to(device).unsqueeze(0)
    features = resnet(img).detach()
    return features


def get_tokenized_question(question):
    words = nltk.word_tokenize(question['question'])
    question_token = []
    for word in words:
        question_token.append(word_dic[word])
    return torch.tensor(question_token).unsqueeze(0)


# +
# plotting
imageDims = (14, 14)
figureImageDims = (2, 3)
figureTableDims = (5, 4)
fontScale = 1

# set transparent mask for low attention areas
# cdict = plt.get_cmap("gnuplot2")._segmentdata
cdict = {
    "red": ((0.0, 0.0, 0.0), (0.6, 0.8, 0.8), (1.0, 1, 1)),
    "green": ((0.0, 0.0, 0.0), (0.6, 0.8, 0.8), (1.0, 1, 1)),
    "blue": ((0.0, 0.0, 0.0), (0.6, 0.8, 0.8), (1.0, 1, 1))
}
cdict["alpha"] = ((0.0, 0.35, 0.35), (1.0, 0.65, 0.65))
plt.register_cmap(name="custom", data=cdict)

def showTableAtt(table, words, tax=None):
    '''
    Question attention as sns heatmap
    '''
    if tax is None:
        fig2, bx = plt.subplots(1, 1)
        bx.cla()
    else:
        bx = tax

    sns.set(font_scale=fontScale)

    steps = len(table)

    # traspose table
    table = np.transpose(table)

    tableMap = pandas.DataFrame(data=table,
                                columns=[i for i in range(1, steps + 1)],
                                index=words)

    bx = sns.heatmap(tableMap,
                     cmap="Purples",
                     cbar=False,
                     linewidths=.5,
                     linecolor="gray",
                     square=True,
                     # ax=bx,
                    )

    # # x ticks
    bx.xaxis.tick_top()
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=0)
    # y ticks
    locs, labels = plt.yticks()
    plt.setp(labels, rotation=0)

# ### Visualizing Image Atts

def showImgAtt(img, atts, step, ax):
    dx, dy = 0.05, 0.05
    x = np.arange(-1.5, 1.5, dx)
    y = np.arange(-1.0, 1.0, dy)
    X, Y = np.meshgrid(x, y)
    extent = np.min(x), np.max(x), np.min(y), np.max(y)

    ax.cla()

    img1 = ax.imshow(img, interpolation="nearest", extent=extent)
    ax.imshow(np.array(atts[step]).reshape(imageDims),
              cmap=plt.get_cmap('custom'),
              interpolation="bicubic",
              extent=extent,
              
             )

    ax.set_axis_off()
    plt.axis("off")

    ax.set_aspect("auto")


def showImgAtts(atts, impath):
    img = imread(impath)

    length = len(atts)

    # show images
    for j in range(length):
        fig, ax = plt.subplots()
        fig.set_figheight(figureImageDims[0])
        fig.set_figwidth(figureImageDims[1])

        showImgAtt(img, atts, j, ax)

        plt.subplots_adjust(bottom=0, top=1, left=0, right=1)


from collections import OrderedDict


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

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
            h = layer.register_forward_hook(hook)
            handles.append(h)
            
        if self.keep_output:
            output = self._model(*args, **kwargs)
        else:
            self._model(*args, **kwargs)
            output = None
            
        for h in handles:
            h.remove()
        
        return ret, output

def setlabel(ax, label, loc=2, borderpad=0.6, **kwargs):
    legend = ax.get_legend()
    if legend:
        ax.add_artist(legend)
    line, = ax.plot(np.NaN,np.NaN,color='none',label=label)
    label_legend = ax.legend(
        handles=[line],
        loc=loc,
        handlelength=0,
        handleheight=0,
        handletextpad=0,
        borderaxespad=0,
        borderpad=borderpad,
        frameon=False,
        prop={
         'size': 18,
         'weight': 'bold',
        },
        **kwargs,
    )
    for text in label_legend.get_texts():
        plt.setp(text, color = 'w')
    label_legend.remove()
    ax.add_artist(label_legend)
    line.remove()
    
def get_image(q_index, split, ds, ds_root):
    image_filename = ds.questions[q_index]['image_filename']
    image = Image.open(os.path.join(ds_root, 'images', split, image_filename)).convert('RGB')
    
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(0.6)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    
    return image

def plot_word_img_attn(
    mid_outputs,
    q_index,
    split,
    ds,
    num_steps,
    ds_root,
    pred,
    gt,
    ):
    fig = plt.figure(figsize=(num_steps, 16))
    
    g0 = gridspec.GridSpec(num_steps // 2, 3, figure=fig)
    
    ax_table = fig.add_subplot(g0[:, 0])
    ax_images = []
    for i in range(num_steps // 2):
        ax_images.append(fig.add_subplot(g0[i, 1]))
        ax_images.append(fig.add_subplot(g0[i, 2]))
    
    table = np.array([t.numpy()[0].squeeze(-1) for t in mid_outputs['cw_attn']])
    steps = len(table)
    table = np.transpose(table)

    # half = math.ceil(len(table) / 2)
    words = nltk.word_tokenize(ds.questions[q_index]['question'])

    tableMap = pandas.DataFrame(data=table,
                                columns=[i for i in range(1, steps + 1)],
                                index=words)

    bx = sns.heatmap(tableMap,
                     cmap="Purples",
                     cbar=True,
                     linewidths=.5,
                     linecolor="gray",
                     square=True,
                     ax=ax_table,
                     cbar_kws={"shrink": .5},
                    )
    bx.set_ylim(bx.get_ylim()[0] + 0.5, bx.get_ylim()[1] - 0.5)
    
    for i in range(num_steps):
        ax = ax_images[i]
        showImgAtt(get_image(q_index, split, ds, ds_root), mid_outputs['kb_attn'], i, ax)
        if i == (num_steps - 1):
            setlabel(ax, f'{pred} ({gt.upper()})')
        else:
            setlabel(ax, str(i + 1))

    plt.tight_layout()
    plt.show()

def idxs_to_question(idxs, mapping):
    return ' '.join([mapping[idx] for idx in idxs])