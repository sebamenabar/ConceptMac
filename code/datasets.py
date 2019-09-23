from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import os.path

import re
import json
import glob
import pickle
import random
from pathlib import Path

import h5py
import numpy as np

import PIL
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from config import cfg


class ClevrDataset(data.Dataset):
    def __init__(self, data_dir, split='train', sample=False):

        self.sample = sample
        if sample:
            sample = '_sample'
        else:
            sample = ''
        with open(os.path.join(data_dir, '{}{}.pkl'.format(split, sample)), 'rb') as f:
            self.data = pickle.load(f)
        # self.img = h5py.File(os.path.join(data_dir, '{}_features.h5'.format(split)), 'r')['features']
        self.img = h5py.File(os.path.join(data_dir, '{}_features.hdf5'.format(split)), 'r')['data']

    def __getitem__(self, index):
        imgfile, question, answer, family = self.data[index]
        id = int(imgfile.rsplit('_', 1)[1][:-4])
        img = torch.from_numpy(self.img[id])

        return img, question, len(question), answer, family

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    images, lengths, answers, _ = [], [], [], []
    batch_size = len(batch)

    max_len = max(map(lambda x: len(x[1]), batch))

    questions = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: len(x[1]), reverse=True)

    for i, b in enumerate(sort_by_len):
        image, question, length, answer, family = b
        images.append(image)
        length = len(question)
        questions[i, :length] = question
        lengths.append(length)
        answers.append(answer)

    return {'image': torch.stack(images), 'question': torch.from_numpy(questions),
            'answer': torch.LongTensor(answers), 'question_length': lengths}

class QOnlyDataset(data.Dataset):
    def __init__(self, data_dir, split='train'):

        with open(os.path.join(data_dir, '{}.pkl'.format(split)), 'rb') as f:
            self.data = pickle.load(f)
        # self.img = h5py.File(os.path.join(data_dir, '{}_features.h5'.format(split)), 'r')['features']

    def __getitem__(self, index):
        imgfile, question, answer, family = self.data[index]
        # id = int(imgfile.rsplit('_', 1)[1][:-4])
        # img = torch.from_numpy(self.img[id])
        img = None

        return img, question, len(question), answer, family

    def __len__(self):
        return len(self.data)


def qonly_collate_fn(batch):
    images, lengths, answers, _ = [], [], [], []
    batch_size = len(batch)

    max_len = max(map(lambda x: len(x[1]), batch))

    questions = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: len(x[1]), reverse=True)

    for i, b in enumerate(sort_by_len):
        image, question, length, answer, family = b
        images.append(image)
        length = len(question)
        questions[i, :length] = question
        lengths.append(length)
        answers.append(answer)

    return {'image': images, 'question': torch.from_numpy(questions),
            'answer': torch.LongTensor(answers), 'question_length': lengths}


class GQADataset(data.Dataset):
    def __init__(self, data_dir, split='train', sample=False):

        self.sample = sample
        if sample:
            sample = '_sample'
        else:
            sample = ''
        with open(os.path.join(data_dir, '{}{}.pkl'.format(split, sample)), 'rb') as f:
            self.data = pickle.load(f)
        with open(os.path.join(data_dir, 'gqa_spatial_merged_info.json')) as f:
            self.spatial_info = json.load(f)
        self.img = h5py.File(os.path.join(data_dir, 'gqa_spatial.h5'), 'r')['features']

    def __getitem__(self, index):
        question = self.data[index]
        imgid, question, answer, group, questionid = self.data[index]
        imgidx = self.spatial_info[imgid]['index']
        img = torch.from_numpy(self.img[imgidx])

        return img, question, len(question), answer, group, questionid, imgid

    def __len__(self):
        return len(self.data)

def collate_fn_gqa(batch):
    images, lengths, answers, _ = [], [], [], []
    batch_size = len(batch)

    max_len = max(map(lambda x: len(x[1]), batch))

    questions = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: len(x[1]), reverse=True)

    for i, b in enumerate(sort_by_len):
        image, question, length, answer, group, qid, imgid = b
        images.append(image)
        length = len(question)
        questions[i, :length] = question
        lengths.append(length)
        answers.append(answer)

    return {'image': torch.stack(images), 'question': torch.from_numpy(questions),
            'answer': torch.LongTensor(answers), 'question_length': lengths}