import os.path as osp
from collections import OrderedDict as odict

from easydict import EasyDict as edict
from pprint import PrettyPrinter as PP

import numpy as np
from skimage.transform import resize

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, resnet101
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter

from base_config import flatten_json_iterative_solution

# from datasets import CLEVRMidrepsDataset, CLEVRMidrepsH5py
from mac import MACNetwork
from encoder import Encoder
from base_pl_model import BasePLModel
from utils import load_vocab
from datasets import ClevrDatasetImages, collate_fn, GQADataset, collate_fn_gqa
from visualize_attentions import plot_vqa_attn

pp = PP(indent=4)


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        return (x - self.mean) / self.std


class PLModel(BasePLModel):
    def __init__(self, cfg=None, hparams=None):
        if cfg is None and hparams is not None:
            cfg = hparams
        super().__init__(cfg)
        self.otf_cfg = edict()

        self.optimizer_groups = odict(group1=[])

        if cfg.model.encoder.type == "resnet50":
            self.init_resnet_encoder(v="resnet50")
        elif cfg.model.encoder.type == "resnet101":
            self.init_resnet_encoder(v="resnet101")
        elif cfg.model.encoder.type == "scratch":
            self.init_scratch_encoder()
        elif cfg.model.encoder.type == "pretrained":
            self.init_pretrained_encoder()
        elif cfg.model.encoder.type == "none":
            self.encoder = nn.Identity()
        else:
            raise ValueError(f"Unkwown encoder {cfg.model.encoder.type}")

        self.init_mac()

        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

    @property
    def dataset_name(self):
        if (self.cfg.train.dataset == "orig") or (self.cfg.train.dataset == "uni"):
            return "clevr"
        elif self.cfg.train.dataset == "gqa":
            return "gqa"

    def __add_param_group(self, group_name, params, module_name):
        default = self.cfg.train.optimizers.default
        if group_name not in self.optimizer_groups:
            self.optimizer_groups[group_name] = []

        options = getattr(self.cfg.train.optimizers, module_name, None)
        self.optimizer_groups[group_name].append(
            {"params": params, "lr": getattr(options, "lr", default.lr),}
        )

    def init_resnet_encoder(self, v):
        if v == "resnet50":
            encoder = resnet50(pretrained=True)
            self.cfg.model.mac.input_unit.in_channels = 1024
        elif v == "resnet101":
            encoder = resnet101(pretrained=True)
            self.cfg.model.mac.input_unit.in_channels = 1024
        encoder = nn.Sequential(*list(encoder.children())[:-3])
        self.encoder = nn.Sequential(
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), encoder,
        )
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    def init_scratch_encoder(self):
        encoder = Encoder(out_nc=self.cfg.model.encoder.out_nc)
        self.encoder = nn.Sequential(
            nn.BatchNorm2d(
                3, affine=False, momentum=False
            ),  # Tricky input normalization
            encoder,
        )
        self.__add_param_group(
            "group1", self.encoder.parameters(), "encoder",
        )

    def init_pretrained_encoder(self):
        encoder = Encoder(out_nc=512)
        ckpt = torch.load(self.cfg.model.encoder.ckpt_fp, map_location="cpu")
        encoder_state_dict = odict(
            {
                k.replace("encoder.", ""): v
                for k, v in ckpt["state_dict"].items()
                if k.startswith("encoder.")
            }
        )
        encoder.load_state_dict(encoder_state_dict, strict=True)
        self.encoder = encoder
        if self.cfg.model.encoder.train:
            self.__add_param_group(
                "group1", self.encoder.parameters(), "encoder",
            )
        else:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

        self.cfg.model.mac.input_unit.in_channels = 512

    def init_mac(self):
        self.vocab = load_vocab(self.cfg.orig_dir)
        kwargs = {
            "vocab": self.vocab,
            "num_answers": len(self.vocab["answer_token_to_idx"]),
        }
        self.mac = MACNetwork(self.cfg.model.mac, **kwargs)
        self.__add_param_group(
            "group1", self.mac.parameters(), "mac",
        )

    def configure_optimizers(self):
        optimizers = [torch.optim.Adam(pgs) for pgs in self.optimizer_groups.values()]
        return optimizers

    def prepare_data(self):
        if self.dataset_name == "clevr":
            self.orig_train_dataset = ClevrDatasetImages(
                base_dir=self.cfg.orig_dir,
                split="train",
                augment=self.cfg.train.augment,
            )

            self.orig_val_dataset = ClevrDatasetImages(
                base_dir=self.cfg.orig_dir, split="val", augment=False,
            )

            self.uni_train_dataset = ClevrDatasetImages(
                base_dir=self.cfg.uni_dir,
                split="train",
                augment=self.cfg.train.augment,
            )

            self.uni_val_dataset = ClevrDatasetImages(
                base_dir=self.cfg.uni_dir, split="val", augment=False,
            )
        else:
            self.train_dataset = GQADataset(
                base_dir=self.cfg.orig_dir,
                split="train",
                augment=self.cfg.train.augment,
                features=self.cfg.model.mac.use_feats,
            )

            self.val_dataset = GQADataset(
                base_dir=self.cfg.orig_dir,
                split="val",
                augment=False,
                features=self.cfg.model.mac.use_feats,
            )

    def train_dataloader(self):
        if self.cfg.train.dataset == "orig":
            dataset = self.orig_train_dataset
            _collate_fn = collate_fn
        elif self.cfg.train.dataset == "uni":
            dataset = self.uni_train_dataset
            _collate_fn = collate_fn
        elif self.cfg.train.dataset == "gqa":
            dataset = self.train_dataset
            _collate_fn = collate_fn_gqa

        return DataLoader(
            dataset,
            shuffle=True,
            drop_last=True,
            batch_size=self.cfg.train.bsz // self.cfg.train.accumulate_grad_batches,
            num_workers=self.cfg.num_workers,
            pin_memory=self.use_cuda,
            collate_fn=_collate_fn,
        )

    def val_dataloader(self):
        if self.dataset_name == "clevr":
            orig_val_loader = DataLoader(
                self.orig_val_dataset,
                shuffle=False,
                drop_last=False,
                batch_size=self.cfg.train.val_bsz,
                num_workers=self.cfg.num_workers,
                pin_memory=self.use_cuda,
                collate_fn=collate_fn,
            )

            uni_val_loader = DataLoader(
                self.uni_val_dataset,
                shuffle=False,
                drop_last=False,
                batch_size=self.cfg.train.val_bsz,
                num_workers=self.cfg.num_workers,
                pin_memory=self.use_cuda,
                collate_fn=collate_fn,
            )

            return [orig_val_loader, uni_val_loader]
        elif self.dataset_name == "gqa":
            val_loader = DataLoader(
                self.val_dataset,
                shuffle=False,
                drop_last=False,
                batch_size=self.cfg.train.val_bsz,
                num_workers=self.cfg.num_workers,
                pin_memory=self.use_cuda,
                collate_fn=collate_fn_gqa,
            )
            return val_loader

    def forward(self, img, question, question_len):
        return self.mac(self.encoder(img), question, question_len)

    def state_dict(self):
        state_dict = super(BasePLModel, self).state_dict()
        if (self.cfg.model.encoder.type == "resnet50") or (
            self.cfg.model.encoder.type == "resnet101"
        ):
            # Dont save resnet weights
            for k in list(state_dict.keys()):
                if k.startswith("encoder"):
                    del state_dict[k]
        return state_dict

    def training_step(self, batch, batch_nb, optimizer_idx=None):
        if batch_nb == 0:
            if self.cfg.train.num_plot_samples > 0:
                self.plot_vqa_attn(
                    batch,
                    "train_step_attn",
                    num_samples=self.cfg.train.num_plot_samples,
                )

        image, question, question_len, answer = (
            batch["image"],
            batch["question"],
            batch["question_length"],
            batch["answer"],
        )
        answer = answer.long()
        pred = self(image, question, question_len)
        loss = self.loss_fn(pred, answer).mean()

        acc = (pred.argmax(1) == answer).float().mean()

        # Gradient clipping done by pytorch lightning
        # if self.cfg.TRAIN.CLIP_GRADS:
        #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.TRAIN.CLIP)

        tqdm_dict = {"acc": acc}

        return {
            "loss": loss,
            "progress_bar": tqdm_dict,
            "log": {"loss": loss, **tqdm_dict},
        }

    def validation_step(self, batch, batch_nb, dataset_idx=None):
        image, question, question_len, answer = (
            batch["image"],
            batch["question"],
            batch["question_length"],
            batch["answer"],
        )
        answer = answer.long()
        pred = self(image, question, question_len)
        loss = self.loss_fn(pred, answer)
        acc = pred.argmax(1) == answer

        # Gradient clipping done by pytorch lightning
        # if self.cfg.TRAIN.CLIP_GRADS:
        #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.TRAIN.CLIP)

        if batch_nb == 0:
            if dataset_idx == 0:
                fig_name = "val_orig_step_attn"
            elif dataset_idx == 1:
                fig_name = "val_uni_step_attn"
            else:
                fig_name = "val_step_attn"
            if self.cfg.train.num_plot_samples > 0:
                self.plot_vqa_attn(
                    batch, fig_name, num_samples=self.cfg.train.num_plot_samples,
                )

        return {
            "loss": loss.cpu(),
            "acc": acc.float().cpu(),
        }

    def validation_epoch_end(self, outputs):
        torch.cuda.empty_cache()

        def mean_agg(olist):
            return torch.cat(olist).float().mean()

        if self.dataset_name == "clevr":
            mean_orig_loss = mean_agg([o["loss"] for o in outputs[0]])
            mean_orig_acc = mean_agg([o["acc"] for o in outputs[0]])
            mean_uni_loss = mean_agg([o["loss"] for o in outputs[1]])
            mean_uni_acc = mean_agg([o["acc"] for o in outputs[1]])

            log = {
                "val_orig_loss": mean_orig_loss,
                "val_orig_acc": mean_orig_acc,
                "val_uni_loss": mean_uni_loss,
                "val_uni_acc": mean_uni_acc,
                "current_epoch": self.current_epoch,
            }
        elif self.dataset_name == "gqa":
            mean_loss = mean_agg([o["loss"] for o in outputs])
            mean_acc = mean_agg([o["acc"] for o in outputs])

            log = {
                "val_loss": mean_loss,
                "val_acc": mean_acc,
                "current_epoch": self.current_epoch,
            }

        self.print()
        self.print(pp.pformat(log))

        return {"log": log}

    def plot_vqa_attn(self, batch, fig_name, num_samples=32, close=True):
        if num_samples > 0:
            with torch.no_grad():
                return_layers = {
                    "mac.mac.control.cw_attn_idty": "cw_attn",
                    "mac.mac.read.kb_attn_idty": "kb_attn",
                }
                mid_getter = MidGetter(self, return_layers, keep_output=True)

                image, question, question_len, answer = (
                    batch["image"],
                    batch["question"],
                    batch["question_length"],
                    batch["answer"],
                )
                answer = answer.long()
                mid_outputs, output = mid_getter(
                    image[:num_samples],
                    question[:num_samples],
                    question_len[:num_samples],
                )
                output = output.detach().cpu()

                for k, v in mid_outputs.items():
                    if isinstance(v, torch.Tensor):
                        mid_outputs[k] = v.detach().cpu()
                    elif isinstance(v, list):
                        mid_outputs[k] = [_v.detach().cpu() for _v in v]

                bsz = question.size(0)
                num_samples = min(num_samples, bsz)

                torch.cuda.empty_cache()

                kb_attn = torch.stack(mid_outputs["kb_attn"], 1).detach().cpu().numpy()
                words_attn = (
                    torch.stack(mid_outputs["cw_attn"], 1)
                    .squeeze(-1)
                    .detach()
                    .cpu()
                    .numpy()
                )

                num_gt_lobs = self.cfg.model.mac.num_gt_lobs
                num_steps = words_attn.shape[1]
                # fig11 = plt.figure(figsize=(16, (bsz * (2 * (num_steps + num_steps // 2) + 4)) // 1))
                fig11 = plt.figure(
                    figsize=(
                        8,
                        (num_samples * (2 * (num_steps + num_steps // 2) + 4)) // 2,
                    )
                )
                outer_grid = fig11.add_gridspec(num_samples, 1, wspace=0.0, hspace=0.1)

                for i in range(num_samples):
                    plot_vqa_attn(
                        img=batch["raw_images"][i],
                        num_steps=num_steps,
                        words=batch["question_words"][i],
                        words_attn=words_attn[i, :, : batch["question_length"][i]],
                        img_attn=kb_attn[i, :, : kb_attn.shape[-1] - num_gt_lobs],
                        prediction=self.vocab["answer_idx_to_token"][
                            output[i].argmax().item()
                        ],
                        real_answer=batch["answer_words"][i],
                        fig=fig11,
                        gridspec=outer_grid[i],
                        num_gt_lobs=num_gt_lobs,
                        gt_lobs_attn=kb_attn[i, :, -num_gt_lobs:],
                    )

                    cw_ax = fig11.get_axes()[i * (3 + num_steps)]
                    cw_ax.set_title(
                        "Question %s" % str(batch["question_idxs"][i]), fontsize=10
                    )
                    img_ax = fig11.get_axes()[i * (3 + num_steps) + 2]
                    # print(ds.questions[q_index])
                    img_ax.set_title(batch["image_fnames"][i], fontsize=6, wrap=True)
                # plt.tight_layout()
                self.log_figure(fig11, fig_name, close=close)

            torch.cuda.empty_cache()

            return fig11
