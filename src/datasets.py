from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import os.path as osp

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

from utils import load_vocab, idxs_to_question

# from config import cfg


class ClevrDatasetImages(data.Dataset):
    def __init__(
        self, base_dir, split="train", augment=False,
    ):
        super().__init__()
        self.base_dir = base_dir
        self.split = split

        if osp.isfile(osp.join(base_dir, "data")):
            info_fpath = osp.join(base_dir, "data", "{}.pkl".format(split))
        else:
            info_fpath = osp.join(base_dir, "features", "{}.pkl".format(split))

        with open(info_fpath, "rb") as f:
            self.data = pickle.load(f)

        # self.aug_transform = transforms.Compose(
        #     [
        #         transforms.RandomPerspective(p=1.0, distortion_scale=0.5),
        #         # transforms.RandomResizedCrop(
        #         #     (224, 224), scale=(0.3, 1), ratio=(0.4, 2)
        #         # ),
        #         # transforms.RandomCrop((256, 256)),
        #         # transforms.Resize((224, 224)),
        #         # transforms.ToTensor(),
        #     ]
        # )
        # self.persp = transforms.RandomPerspective(p=1.0, distortion_scale=0.4)
        # self.rrc = transforms.RandomResizedCrop(
        #     (224, 224), scale=(0.4, 1), ratio=(0.3, 2)
        # )

        # self.rrc = transforms.RandomCrop((256, 256))
        if augment:
            self.augment = transforms.Compose(
                [
                    transforms.RandomPerspective(p=1.0, distortion_scale=0.7,),
                    transforms.RandomCrop((224, 256)),
                    transforms.Resize((224, 224)),
                ]
            )
        else:
            self.augment = None
        self.resize = transforms.Resize((224, 224))
        self.to_tensor = transforms.ToTensor()
        self.vocab = load_vocab(base_dir)

    def get_img_fp(self, img_fname):
        return osp.join(self.base_dir, "images", self.split, img_fname)

    def load_image(self, img_fp):
        return Image.open(img_fp).convert("RGB")

    def get_for_viz(self, index, persp=False, rrc=False):
        img_fname, question, answer, family = self.data[index]
        question_words = idxs_to_question(question, self.vocab["question_idx_to_token"])
        answer_word = self.vocab["answer_idx_to_token"][answer]
        img_fp = self.get_img_fp(img_fname)
        img = self.load_image(img_fp)

        if persp:
            img = self.persp(img)
        if rrc:
            img = self.rrc(img)
        # img = self.resize(img)
        timg = self.to_tensor(img)

        return dict(
            image=timg,
            question=question,
            question_len=len(question),
            answer=answer,
            question_words=question_words,
            answer_word=answer_word,
            raw_image=img,
            question_idx=index,
            image_fname=img_fp,
        )

    def __getitem__(self, index):
        img_fname, question, answer, family = self.data[index]
        question_words = idxs_to_question(question, self.vocab["question_idx_to_token"])
        answer_word = self.vocab["answer_idx_to_token"][answer]
        img_fp = self.get_img_fp(img_fname)
        img = self.load_image(img_fp)
        timg = img
        if self.augment:
            img = self.augment(img)
        else:
            img = self.resize(img)
        timg = self.to_tensor(img)

        return dict(
            image=timg,
            question=question,
            question_len=len(question),
            answer=answer,
            question_words=question_words,
            answer_word=answer_word,
            raw_image=img,
            question_idx=index,
            image_fname=img_fp,
        )

    def __len__(self):
        return len(self.data)


class ClevrDataset(data.Dataset):
    def __init__(
        self,
        data_dir,
        split="train",
        sample=False,
        sample_size=100,
        # Compatibility only
        feats_fname="",
        info_fname="",
        spatial_feats_dset_name="features",
        objects_feats_dset_name="features",
        objects_bboxes_dset_name="bboxes",
    ):

        self.sample = sample
        self.sample_size = sample_size

        info_fpath = os.path.join(data_dir, "{}.pkl".format(split))
        with open(info_fpath, "rb") as f:
            self.data = pickle.load(f)

        if feats_fname:
            feats_fp = os.path.join(data_dir, feats_fname.format(split))
        else:
            feats_fp = os.path.join(data_dir, "{}_features.hdf5".format(split))
        self.img = h5py.File(feats_fp, "r")[spatial_feats_dset_name]

    def __getitem__(self, index):
        imgfile, question, answer, family = self.data[index]
        id = int(imgfile.rsplit("_", 1)[1][:-4])
        img = torch.from_numpy(self.img[id])

        return img, question, len(question), answer, family

    def __len__(self):
        if self.sample:
            return self.sample_size
        return len(self.data)


def collate_fn(batch):
    (
        images,
        lengths,
        answers,
        question_words,
        answer_words,
        raw_images,
        question_idxs,
        image_fnames,
    ) = [[] for _ in range(8)]
    batch_size = len(batch)

    max_len = max(map(lambda x: x["question_len"], batch))

    questions = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: x["question_len"], reverse=True)

    for i, b in enumerate(sort_by_len):
        # image, question, length, answer, family = b
        images.append(b["image"])
        length = b["question_len"]
        questions[i, :length] = b["question"]
        lengths.append(length)
        answers.append(b["answer"])

        question_words.append(b["question_words"])
        answer_words.append(b["answer_word"])
        raw_images.append(b["raw_image"])
        question_idxs.append(b["question_idx"])
        image_fnames.append(b["image_fname"])

    return {
        "image": torch.stack(images),
        "question": torch.from_numpy(questions),
        "answer": torch.LongTensor(answers),
        "question_length": lengths,
        "question_words": question_words,
        "answer_words": answer_words,
        "raw_images": raw_images,
        "question_idxs": question_idxs,
        "image_fnames": image_fnames,
    }


class QOnlyDataset(data.Dataset):
    def __init__(self, data_dir, split="train"):

        with open(os.path.join(data_dir, "{}.pkl".format(split)), "rb") as f:
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

    return {
        "image": images,
        "question": torch.from_numpy(questions),
        "answer": torch.LongTensor(answers),
        "question_length": lengths,
    }


class GQADataset(data.Dataset):
    def __init__(
        self,
        base_dir,
        split="train",
        augment=False,
        features="spatial",
        spatial_feats_dset_name="features",
        objects_feats_dset_name="features",
        objects_bboxes_dset_name="bboxes",
        include_image=False,
    ):
        self.base_dir = base_dir
        self.data_dir = osp.join(base_dir, "data")
        self.split = split
        self.feature_type = features
        self.include_image = include_image
        self.to_tensor = transforms.ToTensor()
        if augment:
            self.augment = transforms.RandomResizedCrop((224, 224))
        else:
            self.augment = transforms.Resize((224, 224))

        with open(osp.join(self.data_dir, f"{split}.pkl"), "rb") as f:
            self.data = pickle.load(f)

        self.vocab = load_vocab(base_dir)

        if self.feature_type == "spatial":
            self.features = h5py.File(osp.join(self.data_dir, "gqa_spatial.h5"), "r")
            self.features = self.features[spatial_feats_dset_name]
            with open(
                osp.join(self.data_dir, "gqa_spatial_merged_info.json"), "r"
            ) as f:
                self.info = json.load(f)

            self.__getitem__ = self.get_spatial

        elif self.feature_type == "objects":
            self.features = h5py.File(osp.join(self.data_dir, "gqa_objects.h5"), "r")
            self.features, self.bboxes = (
                self.features[objects_feats_dset_name],
                self.features[objects_bboxes_dset_name],
            )
            with open(
                osp.join(self.data_dir, "gqa_objects_merged_info.json"), "r"
            ) as f:
                self.info = json.load(f)

            self.__getitem__ = self.get_objects

        elif self.feature_type == "pixels":
            self.__getitem__ = self.get_pixels

    def __getitem__(self, index, feature_type=None):
        if feature_type is None:
            feature_type = self.feature_type

        imgid, question, answer, group, questionid = self.data[index]

        out = dict(
            #  image=features,
            question=question,
            question_len=len(question),
            answer=answer,
            question_words=idxs_to_question(
                question, self.vocab["question_idx_to_token"]
            ),
            answer_word=self.vocab["answer_idx_to_token"][answer],
            # raw_image=self.open_image(imgid, (224, 224)),
            group=group,
            question_idx=questionid,
            image_fname=imgid,
        )

        if feature_type == "spatial":
            img_info = self.info[imgid]
            imgidx = img_info["index"]
            out["image"] = self.get_spatial(imgidx)
            out["raw_image"] = self.open_image(imgid, (224, 224))
        elif feature_type == "objects":
            img_info = self.info[imgid]
            imgidx = img_info["index"]
            out["image"] = self.get_objects(imgidx, img_info)
            out["raw_image"] = self.open_image(imgid, (224, 224))
        elif feature_type == "pixels":
            timg, img = self.get_pixels(imgid)
            out["image"] = timg
            out["raw_image"] = img
        else:
            raise AttributeError(f"Unknown feature type {self.feature_type}")

        return out

    def __len__(self):
        return len(self.data)

    def open_image(self, imgid, resize=None):
        img = Image.open(osp.join(self.base_dir, "images", f"{imgid}.jpg")).convert(
            "RGB"
        )
        if resize:
            img = transforms.functional.resize(img, resize)
        return img

    def get_spatial(self, imgidx):
        return torch.from_numpy(self.features[imgidx])

    def get_pixels(self, imgid):
        img = self.open_image(imgid)
        img = self.augment(img)
        return self.to_tensor(img), img

    def get_objects(self, imgidx, img_info):
        h, w = img_info["height"], img_info["width"]
        bboxes = self.bboxes[imgidx] / (w, h, w, h)
        img = self.features[imgidx]

        bboxes = bboxes[: img_info["objectsNum"]]
        img = img[: img_info["objectsNum"]]
        img = torch.from_numpy(np.concatenate((img, bboxes), axis=1).astype(np.float32))
        img = (img, img_info["objectsNum"])
        return img


def collate_fn_gqa(batch):
    (
        images,
        lengths,
        answers,
        question_words,
        answer_words,
        raw_images,
        question_idxs,
        image_fnames,
    ) = [[] for _ in range(8)]
    batch_size = len(batch)

    max_len = max(map(lambda x: x["question_len"], batch))

    questions = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: x["question_len"], reverse=True)

    obj_lengths = []
    for i, b in enumerate(sort_by_len):
        image = b["image"]
        if len(image) == 2:
            images.append(image[0])
            obj_lengths.append(image[1])
        else:
            images.append(image)
        length = b["question_len"]
        questions[i, :length] = b["question"]
        lengths.append(length)
        answers.append(b["answer"])

        question_words.append(b["question_words"])
        answer_words.append(b["answer_word"])
        raw_images.append(b["raw_image"])
        question_idxs.append(b["question_idx"])
        image_fnames.append(b["image_fname"])

    if obj_lengths:
        _img_out = (
            torch.nn.utils.rnn.pad_sequence(images, batch_first=True),
            torch.as_tensor(obj_lengths),
        )
    else:
        _img_out = torch.stack(images)

    return {
        "image": _img_out,
        "question": torch.from_numpy(questions),
        "answer": torch.LongTensor(answers),
        "question_length": lengths,
        "question_words": question_words,
        "answer_words": answer_words,
        "raw_images": raw_images,
        "question_idxs": question_idxs,
        "image_fnames": image_fnames,
    }


# class GQADataset(data.Dataset):
#     def __init__(
#         self,
#         data_dir,
#         split="train",
#         sample=False,
#         use_feats="spatial",
#         feats_fname="",
#         info_fname="",
#         spatial_feats_dset_name="features",
#         objects_feats_dset_name="features",
#         objects_bboxes_dset_name="bboxes",
#     ):

#         self.use_feats = use_feats
#         self.sample = sample
#         if sample:
#             sample = "_sample"
#         else:
#             sample = ""

#         with open(os.path.join(data_dir, "{}{}.pkl".format(split, sample)), "rb") as f:
#             self.data = pickle.load(f)

#         if feats_fname == "":
#             feats_fname = f"gqa_{self.use_feats}.h5"
#         if info_fname == "":
#             info_fname = f"gqa_{self.use_feats}_merged_info.json"
#         feats_fname = os.path.join(data_dir, feats_fname)
#         info_fname = os.path.join(data_dir, info_fname)

#         self.features = h5py.File(feats_fname, "r")
#         with open(os.path.join(data_dir, info_fname), "r") as f:
#             self.info = json.load(f)

#         if self.use_feats == "spatial":
#             self.features = self.features[spatial_feats_dset_name]
#         elif self.use_feats == "objects":
#             self.features, self.bboxes = (
#                 self.features[objects_feats_dset_name],
#                 self.features[objects_bboxes_dset_name],
#             )

#     def __getitem__(self, index):
#         imgid, question, answer, group, questionid = self.data[index]
#         img_info = self.info[imgid]
#         imgidx = img_info["index"]

#         if self.use_feats == "spatial":
#             img = torch.from_numpy(self.features[imgidx])
#         elif self.use_feats == "objects":
#             h, w = img_info["height"], img_info["width"]
#             bboxes = self.bboxes[imgidx] / (w, h, w, h)
#             img = self.features[imgidx]

#             bboxes = bboxes[: img_info["objectsNum"]]
#             img = img[: img_info["objectsNum"]]

#             # img = torch.from_numpy(np.concatenate((img, bboxes), axis=1)).to(torch.float32)
#             img = torch.from_numpy(
#                 np.concatenate((img, bboxes), axis=1).astype(np.float32)
#             )
#             img = (img, img_info["objectsNum"])

#         return img, question, len(question), answer, group, questionid, imgid

#     def __len__(self):
#         return len(self.data)


# def collate_fn_gqa_old(batch):
#     images, lengths, answers, _ = [], [], [], []
#     batch_size = len(batch)

#     max_len = max(map(lambda x: len(x[1]), batch))

#     questions = np.zeros((batch_size, max_len), dtype=np.int64)
#     sort_by_len = sorted(batch, key=lambda x: len(x[1]), reverse=True)

#     for i, b in enumerate(sort_by_len):
#         image, question, length, answer, group, qid, imgid = b
#         images.append(image)
#         length = len(question)
#         questions[i, :length] = question
#         lengths.append(length)
#         answers.append(answer)

#     return {
#         "image": torch.stack(images),
#         "question": torch.from_numpy(questions),
#         "answer": torch.LongTensor(answers),
#         "question_length": lengths,
#     }


# def collate_fn_gqa_objs_old(batch):
#     images, obj_lengths, lengths, answers = [], [], [], []
#     batch_size = len(batch)

#     max_len = max(map(lambda x: len(x[1]), batch))

#     questions = np.zeros((batch_size, max_len), dtype=np.int64)
#     sort_by_len = sorted(batch, key=lambda x: len(x[1]), reverse=True)

#     for i, b in enumerate(sort_by_len):
#         (image, obj_length), question, length, answer, group, qid, imgid = b
#         images.append(image)
#         obj_lengths.append(obj_length)
#         length = len(question)
#         questions[i, :length] = question
#         lengths.append(length)
#         answers.append(answer)

#     # images = torch.stack(images)
#     images = torch.nn.utils.rnn.pad_sequence(images, batch_first=True)

#     return {
#         "image": (
#             # torch.nn.utils.rnn.pad_sequence(images, batch_first=True),
#             images,
#             torch.as_tensor(obj_lengths),
#         ),
#         "question": torch.from_numpy(questions),
#         "answer": torch.LongTensor(answers),
#         "question_length": lengths,
#     }
