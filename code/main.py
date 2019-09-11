from __future__ import print_function

import os
import sys

import random
import shutil
import argparse
import datetime
import dateutil
import dateutil.tz

import comet_ml
from dotenv import load_dotenv

import torch

# dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
# sys.path.append(dir_path)

from utils import mkdir_p
from trainer import Trainer
from config import cfg, cfg_from_file

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default=None, type=str)
    parser.add_argument('--gpu',  dest='gpu_id', type=str, default='0')
    parser.add_argument('--data-dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--lobs', type=int, help='numbers of learnable objects')
    parser.add_argument('--cogent', type=str)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--resume-model', type=str)
    parser.add_argument('--resume-model-ema', type=str)
    parser.add_argument('--bsz', type=int)
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--start-epoch', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--logcomet', action='store_true')
    args = parser.parse_args()
    return args


def set_logdir(max_steps, lobs, logdir=None):
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    if logdir is None:
        logdir = "data/{}_max_steps_{}_lobs_{}".format(now, max_steps, lobs)
    else:
        logdir = f'data/{logdir}'
    mkdir_p(logdir)
    print("Saving output to: {}".format(logdir))
    code_dir = os.path.join(os.getcwd(), "code")
    mkdir_p(os.path.join(logdir, "Code"))
    for filename in os.listdir(code_dir):
        if filename.endswith(".py"):
            shutil.copy(code_dir + "/" + filename, os.path.join(logdir, "Code"))
    shutil.copy(args.cfg_file, logdir)
    return logdir


if __name__ == "__main__":
    args = parse_args()
    cfg.logcomet = args.logcomet
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    if args.gpu_id == '-1':
        cfg.CUDA = False
    if args.data_dir != '':
        cfg.DATASET.DATA_DIR = args.data_dir
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    if args.lobs is not None:
        cfg.model.num_lobs = args.lobs
    if args.cogent is not None:
        cfg.DATASET.COGENT = args.cogent.upper()
    if args.resume_model is not None:
        cfg.resume_model = args.resume_model
        cfg.resume_model_ema = args.resume_model_ema
    if args.bsz is not None:
        cfg.TEST_BATCH_SIZE = args.bsz
        cfg.TRAIN.BATCH_SIZE = args.bsz
    if args.start_epoch is not None:
        cfg.start_epoch = args.start_epoch
    if args.epochs is not None:
        cfg.TRAIN.MAX_EPOCHS = args.epochs
    if args.workers is not None:
        cfg.WORKERS = args.workers
    if args.logdir:
        cfg.LOGDIR = args.logdir

    cfg.exp_name = cfg.LOGDIR

    cfg.SAMPLE = args.sample
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    if args.eval or args.test:
        cfg.TRAIN.FLAG = False
    if args.test:
        cfg.TEST = True
        args.EVAL = False
    cfg.EVAL = args.eval
    print(cfg)

    logdir = set_logdir(cfg.model.max_step, cfg.model.num_lobs, cfg.LOGDIR)
    trainer = Trainer(logdir, cfg)

    if cfg.TRAIN.FLAG:
        trainer.train()
    elif cfg.EVAL or cfg.TEST:
        mode = 'validation' if cfg.EVAL else 'test'
        trainer.comet_exp.disable_mp()
        metrics = trainer.calc_accuracy(mode)

        print('Acc: {:.4f}'.format(metrics['acc']))
        print('Acc EMA: {:.4f}'.format(metrics['acc_ema']))

        print('Loss: {:.4f}'.format(metrics['loss']))
        print('Loss EMA: {:.4f}'.format(metrics['loss_ema']))

    # elif cfg.TEST:
    #     accuracy, accuracy_ema = trainer.calc_accuracy('test')

        # raise NotImplementedError

