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
    
    # Config
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default=None, type=str)
    parser.add_argument('--gpu',  dest='gpu_id', type=str, default='0')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--manualSeed', type=int, help='manual seed')

    # Resume
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--start-epoch', type=int)
    parser.add_argument('--resume-model', type=str)
    parser.add_argument('--resume-model-ema', type=str)

    # Logs
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--comet-project-name', type=str)
    parser.add_argument('--logcomet', action='store_true')


    # Optimizer
    parser.add_argument('--bsz', type=int)
    parser.add_argument('--lr', type=float)

    # Data
    parser.add_argument('--cogent', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--train_split', type=str)
    parser.add_argument('--eval', default='')
    # parser.add_argument('--test', action='store_true')
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--data-dir', dest='data_dir', type=str, default='')

    args = parser.parse_args()
    return args


def set_logdir(max_steps, logdir='', eval=''):
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    if logdir:
        logdir = f'data/{logdir}'
    else:
        logdir = "data/{}_max_steps_{}".format(now, max_steps)
    if eval:
        logdir += f'_eval-{eval}'
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
        cfg.cfg_file = args.cfg_file
        cfg_from_file(args.cfg_file)
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    if args.gpu_id == '-1':
        cfg.CUDA = False
    if args.data_dir != '':
        cfg.DATASET.DATA_DIR = args.data_dir
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    if args.cogent is not None:
        cfg.DATASET.COGENT = args.cogent.upper()
    if args.resume_model is not None:
        cfg.resume_model = args.resume_model
        cfg.resume_model_ema = args.resume_model_ema
    if args.bsz is not None:
        cfg.TEST_BATCH_SIZE = args.bsz
        cfg.TRAIN.BATCH_SIZE = args.bsz
    if args.lr is not None:
        cfg.TRAIN.LEARNING_RATE = args.lr
    if args.start_epoch is not None:
        cfg.start_epoch = args.start_epoch
    if args.epochs is not None:
        cfg.TRAIN.MAX_EPOCHS = args.epochs
    if args.workers is not None:
        cfg.WORKERS = args.workers
    if args.logdir:
        cfg.LOGDIR = args.logdir
    if args.dataset is not None:
        cfg.DATASET.DATASET = args.dataset
    if args.train_split is not None:
        cfg.DATASET.train_split = args.train_split

    if args.comet_project_name is not None:
        cfg.COMET_PROJECT_NAME = args.comet_project_name
    elif cfg.COMET_PROJECT_NAME is None:
        cfg.COMET_PROJECT_NAME = os.getenv('COMET_PROJECT_NAME')

    cfg.exp_name = cfg.LOGDIR

    cfg.SAMPLE = args.sample
    cfg.manualSeed = args.manualSeed
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    if args.eval:
        cfg.TRAIN.FLAG = False
        cfg.logcomet = False
    cfg.EVAL = args.eval

    logdir = set_logdir(cfg.model.max_step, cfg.LOGDIR, eval=cfg.EVAL)
    trainer = Trainer(logdir, cfg)

    if cfg.TRAIN.FLAG:
        trainer.train()
    elif cfg.EVAL:
        mode = cfg.EVAL
        trainer.comet_exp.disable_mp()
        metrics = trainer.calc_accuracy(mode)

        print('Acc: {:.4f}'.format(metrics['acc']))
        print('Acc EMA: {:.4f}'.format(metrics['acc_ema']))

        print('Loss: {:.4f}'.format(metrics['loss']))
        print('Loss EMA: {:.4f}'.format(metrics['loss_ema']))
