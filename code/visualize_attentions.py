#!/usr/bin/env python
# coding: utf-8

from collections import OrderedDict
import os
import sys

import math
import functools

import cv2
import nltk
import pandas
import numpy as np
from PIL import Image, ImageEnhance

import torch
from torch import nn

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LinearSegmentedColormap


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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def showImgAtt(img, atts, step, ax):
    dx, dy = 0.05, 0.05
    x = np.arange(-1.5, 1.5, dx)
    y = np.arange(-1.0, 1.0, dy)
    X, Y = np.meshgrid(x, y)
    extent = np.min(x), np.max(x), np.min(y), np.max(y)

    ax.cla()

    dim = int(math.sqrt(len(atts[0][0])))
    img1 = ax.imshow(img, interpolation="nearest", extent=extent)

    att = atts[step][0]

    low = att.min().item()
    high = att.max().item()
    att = sigmoid(((att - low) / (high - low)) * 20 - 10)

    ax.imshow(att.reshape((dim, dim)),
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
    line, = ax.plot(np.NaN, np.NaN, color='none', label=label)
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
        plt.setp(text, color='w')
    label_legend.remove()
    ax.add_artist(label_legend)
    line.remove()


def get_image(image_path, enhance=True):
    image = Image.open(image_path).convert('RGB')

    if enhance:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(0.5)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.6)

    return image

def plot_word_img_attn(
        mid_outputs,
        num_steps,
        words,
        images_root,
        image_filename,
        pred,
        gt,
    ):
    fig = plt.figure(figsize=(16, 2 * num_steps + 4))

    g0 = gridspec.GridSpec((num_steps // 2) + 2, 3, figure=fig)

    ax_raw_image = fig.add_subplot(g0[-2:, 1:])
    image_path = os.path.join(images_root, image_filename)
    img = image = Image.open(image_path).convert('RGB')
    ax_raw_image.imshow(img)
    ax_raw_image.set_axis_off()
    # ax_raw_image.set_aspect("auto")

    ax_table = fig.add_subplot(g0[:-1, 0])
    ax_images = []
    for i in range(num_steps // 2):
        ax_images.append(fig.add_subplot(g0[i, 1]))
        ax_images.append(fig.add_subplot(g0[i, 2]))

    table = np.array([t.numpy()[0].squeeze(-1)
                      for t in mid_outputs['cw_attn']])
    steps = len(table)
    table = np.transpose(table)

    tableMap = pandas.DataFrame(data=table,
                                columns=[i for i in range(1, steps + 1)],
                                index=words)

    bx = sns.heatmap(tableMap,
                     cmap="Greys",
                     cbar=True,
                     linewidths=.5,
                     linecolor="gray",
                     square=True,
                     ax=ax_table,
                     cbar_kws={"shrink": .5},
                     )
    bx.set_ylim(bx.get_ylim()[0] + 0.5, bx.get_ylim()[1] - 0.5)
    bx.set_yticklabels(bx.get_yticklabels(), rotation = 0, fontsize = 15)


    for i in range(num_steps):
        ax = ax_images[i]
        showImgAtt(get_image(image_path),
                   mid_outputs['kb_attn'], i, ax)
        if i == (num_steps - 1):
            setlabel(ax, f'{pred} ({gt.upper()})')
        else:
            setlabel(ax, str(i + 1))

    plt.tight_layout()
    plt.show()

def interpolate(val, x_low, x_high):
    return (val - x_low) / (x_high - x_low)

def plot_word_img_attn_objs(
        mid_outputs,
        num_steps,
        words,
        images_root,
        image_filename,
        pred,
        gt,
        num_gt_objs,
        bboxes,
        num_gt_lobs=0,
    ):
    fig = plt.figure(figsize=(16, 2 * num_steps + 4))

    grid_h = (num_steps // 2) + 2
    g0 = gridspec.GridSpec(grid_h, 3, figure=fig)

    ax_raw_image = fig.add_subplot(g0[-2:, 1:])
    image_path = os.path.join(images_root, image_filename)
    img = np.array(Image.open(image_path).convert('RGB'))
    ax_raw_image.imshow(img)
    ax_raw_image.set_axis_off()

    ax_table_cw = fig.add_subplot(g0[:math.ceil(grid_h / 2), 0])
    ax_table_objs = fig.add_subplot(g0[math.ceil(grid_h / 2):, 0])

    table_cw = np.array([t.numpy()[0].squeeze(-1) for t in mid_outputs['cw_attn']])
    steps = len(table_cw)
    table_cw = np.transpose(table_cw)
    # words = nltk.word_tokenize(ds.questions[q_index]['question'])
    tableMap = pandas.DataFrame(data=table_cw,
                                columns=[i for i in range(1, steps + 1)],
                                index=words)
    bx = sns.heatmap(tableMap,
                    cmap="Greys",
                    cbar=True,
                    linewidths=.5,
                    linecolor="gray",
                    square=True,
                    ax=ax_table_cw,
                    cbar_kws={"shrink": .5},
                    )
    bx.set_ylim(bx.get_ylim()[0] + 0.5, bx.get_ylim()[1] - 0.5)
    bx.set_yticklabels(bx.get_yticklabels(), rotation = 0, fontsize = 15)


    ax_images = []
    for i in range(num_steps // 2):
        ax_images.append(fig.add_subplot(g0[i, 1]))
        ax_images.append(fig.add_subplot(g0[i, 2]))

    table_cw = np.array([t.numpy()[0].squeeze(-1) for t in mid_outputs['cw_attn']])
    steps = len(table_cw)
    table_cw = np.transpose(table_cw)
    # words = nltk.word_tokenize(ds.questions[q_index]['question'])
    tableMap = pandas.DataFrame(data=table_cw,
                                columns=[i for i in range(1, steps + 1)],
                                index=words)

    objs_attn = torch.cat(mid_outputs['kb_attn'])
    objs_attn = objs_attn.t().numpy()
    tableMap = pandas.DataFrame(data=objs_attn,
                                columns=[i for i in range(1, num_steps + 1)],
                                )
    bx = sns.heatmap(tableMap,
                    cmap="Greys",
                    cbar=True,
                    linewidths=.5,
                    linecolor="gray",
                    square=True,
                    ax=ax_table_objs,
                    yticklabels=['Obj %d' % i for i in range(num_gt_objs)] + ['LObj %d' % i for i in range(1, num_gt_lobs + 1)],
                    cbar_kws={"shrink": .5},
                    # vmin=0, vmax=1,
                    )
    bx.set_ylim(bx.get_ylim()[0] + 0.5, bx.get_ylim()[1] - 0.5)
    bx.xaxis.set_ticks_position('top')
    bx.set_yticklabels(bx.get_yticklabels(), rotation = 0, fontsize = 12)


    h, w = img.shape[0], img.shape[1]
    for i in range(num_steps):
        img_i = img.copy()
        ax_i = ax_images[i]

        gt_obbs_attn_i = mid_outputs['kb_attn'][i][0]
        low, high = gt_obbs_attn_i.min().item(), gt_obbs_attn_i.max().item()
        for j in range(num_gt_objs):
            box_attn_ij = gt_obbs_attn_i[j].item()
            box_abs_coords_j = bboxes[j] * (w, h, w, h)
            top_left, bottom_right = box_abs_coords_j[:2].astype(np.int64).tolist(), box_abs_coords_j[2:].astype(np.int64).tolist()

            score = interpolate(box_attn_ij, low, high)
            c_intensity = 255 * score
            linewidth = 2 + (4 * score)

            img_i = cv2.rectangle(img_i, tuple(top_left), tuple(bottom_right), (0, math.ceil(c_intensity), 0), int(round(linewidth)))

        ax_i.imshow(img_i)
        if i == (num_steps - 1):
            setlabel(ax_i, f'{pred} ({gt.upper()})')
        else:
            setlabel(ax_i, str(i + 1))

        ax_i.set_axis_off()
        ax_i.set_aspect("auto")

    # plt.tight_layout()
    # plt.show()

    return fig

def plot_word_img_attn_lobs(
        mid_outputs,
        num_steps,
        words,
        images_root,
        image_filename,
        pred,
        gt,
        num_lobs=0,
        read_gate=False,
    ):
    fig = plt.figure(figsize=(16, 2 * num_steps + 4))

    grid_h = (num_steps // 2) + 2
    g0 = gridspec.GridSpec(grid_h, 3, figure=fig)

    ax_raw_image = fig.add_subplot(g0[-2:, 1:])
    image_path = os.path.join(images_root, image_filename)
    img = image = Image.open(image_path).convert('RGB')
    ax_raw_image.imshow(img)
    ax_raw_image.set_axis_off()
    # ax_raw_image.set_aspect("auto")

    # quart = math.ceil(len(table) / 4)
    ax_table_cw = fig.add_subplot(g0[:math.ceil(grid_h / 2), 0])
    ax_table_objs = fig.add_subplot(g0[math.ceil(grid_h / 2):, 0])

    ax_images = []
    for i in range(num_steps // 2):
        ax_images.append(fig.add_subplot(g0[i, 1]))
        ax_images.append(fig.add_subplot(g0[i, 2]))

    table_cw = np.array([t.numpy()[0].squeeze(-1) for t in mid_outputs['cw_attn']])
    steps = len(table_cw)
    table_cw = np.transpose(table_cw)
    # words = nltk.word_tokenize(ds.questions[q_index]['question'])
    tableMap = pandas.DataFrame(data=table_cw,
                                columns=[i for i in range(1, steps + 1)],
                                index=words)
    bx = sns.heatmap(tableMap,
                     cmap="Greys",
                     cbar=True,
                     linewidths=.5,
                     linecolor="gray",
                     square=True,
                     ax=ax_table_cw,
                     cbar_kws={"shrink": .5},
                     )
    bx.set_ylim(bx.get_ylim()[0] + 0.5, bx.get_ylim()[1] - 0.5)
    bx.set_yticklabels(bx.get_yticklabels(), rotation = 0, fontsize = 15)

    for i in range(num_steps):
        ax = ax_images[i]
        showImgAtt(get_image(os.path.join(images_root, image_filename)),
                   mid_outputs['kb_attn'], i, ax)
        if i == (num_steps - 1):
            setlabel(ax, f'{pred} ({gt.upper()})')
        else:
            setlabel(ax, str(i + 1))

    if num_lobs > 0 and read_gate:
        num_lobs = mid_outputs['lobs_attn'][0].size(0)
        lobs_attn = torch.cat(mid_outputs['lobs_attn']) * (1 - torch.cat(mid_outputs['read_gate']))
        attn = torch.cat([torch.cat(mid_outputs['read_gate']), lobs_attn], dim=1)
    # elif read_gate:
    #     attn = mid_outputs['read_gate']
        attn = attn.t().numpy()
        tableMap = pandas.DataFrame(data=attn,
                                    columns=[i for i in range(1, num_steps + 1)],
                                )
        bx = sns.heatmap(tableMap,
                        cmap="Greys",
                        cbar=True,
                        linewidths=.5,
                        linecolor="gray",
                        square=True,
                        ax=ax_table_objs,
                        yticklabels=['KB'] + ['LObj %d' % i for i in range(1, num_lobs + 1)],
                        cbar_kws={"shrink": .5},
                        vmin=0, vmax=1,
                        )
        bx.set_ylim(bx.get_ylim()[0] + 0.5, bx.get_ylim()[1] - 0.5)
        bx.xaxis.set_ticks_position('top')

        bx.set_yticklabels(bx.get_yticklabels(), rotation = 0, fontsize = 12)
    else:
        ax_table_objs.set_visible(False)
        ax_table_objs.set_axis_off()
        ax_table_objs.cla()

    plt.tight_layout()
    plt.show()

def idxs_to_question(idxs, mapping):
    return [mapping[idx] for idx in idxs]
