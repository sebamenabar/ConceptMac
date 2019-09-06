from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import PIL
import os
import os.path
import pickle
import random
import numpy as np
import glob
from pathlib import Path
import json
import h5py
import re
import random

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

classes = {
            'number':['0','1','2','3','4','5','6','7','8','9','10'],
            'material':['rubber','metal'],
            'color':['cyan','blue','yellow','purple','red','green','gray','brown'],
            'shape':['sphere','cube','cylinder'],
            'size':['large','small'],
            'exist':['yes','no']
        }

class ClevrScenesDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, scenes_dir, split='train', sample=False):

        self.sample = sample
        if sample:
            sample = '_sample'
        else:
            sample = ''
        with open(os.path.join(data_dir, '{}{}.pkl'.format(split, sample)), 'rb') as f:
            self.data = pickle.load(f)
        with open(os.path.join(scenes_dir, f'CLEVR_{split}_scenes.json'), 'r') as f:
            scenes = json.load(f)['scenes']
        # self.img = h5py.File(os.path.join(data_dir, '{}_features.h5'.format(split)), 'r')['features']
        # self.img = h5py.File(os.path.join(data_dir, '{}_features.hdf5'.format(split)), 'r')['data']

        self.all_objects = []
        for s in scenes:
            objects = s['objects']
            objects_attr = []
            for obj in objects:
                attr_values = []
                for attr in sorted(obj):
                    # convert object attributes in indexes
                    if attr in classes:
                        attr_values.append(classes[attr].index(obj[attr]))  #zero is reserved for padding
                    else:
                        # if attr=='rotation':
                        #     attr_values.append(float(obj[attr]) / 360)
                        if attr=='3d_coords':
                            attr_values.extend(obj[attr])
                objects_attr.append(attr_values)
            self.all_objects.append(objects_attr)
        
        
    def __getitem__(self, index):
        imgfile, question, answer, family = self.data[index]
        id = int(imgfile.rsplit('_', 1)[1][:-4])
        # img = torch.from_numpy(self.img[id])
        scene = self.all_objects[id]
                
        return scene, question, len(question), answer, family

    def __len__(self):
        return len(self.data)

def scenes_collate_fn(batch):
    scenes, lengths, answers, _ = [], [], [], []
    batch_size = len(batch)

    max_len = max(map(lambda x: len(x[1]), batch))

    questions = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: len(x[1]), reverse=True)

    for i, b in enumerate(sort_by_len):
        scene, question, length, answer, family = b
        scene = torch.FloatTensor(scene)
        scenes.append(scene)
        length = len(question)
        questions[i, :length] = question
        lengths.append(length)
        answers.append(answer)

    scenes_len = [len(s) for s in scenes]
    scenes = torch.nn.utils.rnn.pack_sequence(scenes, enforce_sorted=False, )
                
    return {'scenes': scenes, 'question': torch.from_numpy(questions),
            'scene_length': torch.as_tensor(scenes_len, dtype=torch.float32),
            'answer': torch.LongTensor(answers), 'question_length': lengths}