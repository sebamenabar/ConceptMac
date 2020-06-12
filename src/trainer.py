from __future__ import print_function

import os
import sys

import json
import shutil
from six.moves import range

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import pprint
from tqdm import tqdm
from comet_ml import Experiment
from tensorboardX import SummaryWriter

import mac as mac
from radam import RAdam
from datasets import ClevrDataset, collate_fn, GQADataset, collate_fn_gqa, collate_fn_gqa_objs
from utils import mkdir_p, save_model, load_vocab, cfg_to_exp_name, flatten_json_iterative_solution


class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        sys.stdout.flush()
        self.log.flush()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


class Trainer():
    def __init__(self, log_dir, cfg):

        self.path = log_dir
        self.cfg = cfg

        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(self.path, 'Model')
            self.log_dir = os.path.join(self.path, 'Log')
            mkdir_p(self.model_dir)
            mkdir_p(self.log_dir)
            self.writer = SummaryWriter(log_dir=self.log_dir)
            self.logfile = os.path.join(self.path, "logfile.log")
            sys.stdout = Logger(logfile=self.logfile)

        self.data_dir = cfg.DATASET.DATA_DIR
        self.max_epochs = cfg.TRAIN.MAX_EPOCHS
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.lr = cfg.TRAIN.LEARNING_RATE

        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True

        sample = cfg.SAMPLE
        self.dataset = []
        self.dataloader = []
        self.use_feats = cfg.model.use_feats
        eval_split = cfg.EVAL if cfg.EVAL else 'val'
        train_split = cfg.DATASET.train_split
        if cfg.DATASET.DATASET == 'clevr':
            clevr_collate_fn = collate_fn
            cogent = cfg.DATASET.COGENT
            if cogent:
                print(f'Using CoGenT {cogent.upper()}')

            if cfg.TRAIN.FLAG:
                self.dataset = ClevrDataset(data_dir=self.data_dir, split=train_split + cogent, sample=sample, **cfg.DATASET.params)
                self.dataloader = DataLoader(dataset=self.dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                                            num_workers=cfg.WORKERS, drop_last=True, collate_fn=clevr_collate_fn)
            
            self.dataset_val = ClevrDataset(data_dir=self.data_dir, split=eval_split + cogent, sample=sample, **cfg.DATASET.params)
            self.dataloader_val = DataLoader(dataset=self.dataset_val, batch_size=cfg.TEST_BATCH_SIZE, drop_last=False,
                                            shuffle=False, num_workers=cfg.WORKERS, collate_fn=clevr_collate_fn)

        elif cfg.DATASET.DATASET == 'gqa':
            if self.use_feats == 'spatial':
                gqa_collate_fn = collate_fn_gqa
            elif self.use_feats == 'objects':
                gqa_collate_fn = collate_fn_gqa_objs
            if cfg.TRAIN.FLAG:
                self.dataset = GQADataset(data_dir=self.data_dir, split=train_split, sample=sample, use_feats=self.use_feats, **cfg.DATASET.params)
                self.dataloader = DataLoader(dataset=self.dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                                            num_workers=cfg.WORKERS, drop_last=True, collate_fn=gqa_collate_fn)
            
            self.dataset_val = GQADataset(data_dir=self.data_dir, split=eval_split, sample=sample, use_feats=self.use_feats, **cfg.DATASET.params)
            self.dataloader_val = DataLoader(dataset=self.dataset_val, batch_size=cfg.TEST_BATCH_SIZE, shuffle=False,
                                        num_workers=cfg.WORKERS, drop_last=False, collate_fn=gqa_collate_fn)

        # load model
        self.vocab = load_vocab(cfg)
        self.model, self.model_ema = mac.load_MAC(cfg, self.vocab)
            
        self.weight_moving_average(alpha=0)
        if cfg.TRAIN.RADAM:
            self.optimizer = RAdam(self.model.parameters(), lr=self.lr)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.start_epoch = 0
        if cfg.resume_model:
            location = 'cuda' if cfg.CUDA else 'cpu'
            state = torch.load(cfg.resume_model, map_location=location)
            self.model.load_state_dict(state['model'])
            self.optimizer.load_state_dict(state['optim'])
            self.start_epoch = state['iter'] + 1
            state = torch.load(cfg.resume_model_ema, map_location=location)
            self.model_ema.load_state_dict(state['model'])

        if cfg.start_epoch is not None:
            self.start_epoch = cfg.start_epoch

        self.previous_best_acc = 0.0
        self.previous_best_epoch = 0
        self.previous_best_loss = 100
        self.previous_best_loss_epoch = 0

        self.total_epoch_loss = 0
        self.prior_epoch_loss = 10

        self.print_info()
        self.loss_fn = torch.nn.CrossEntropyLoss().cuda()

        self.comet_exp = Experiment(
            project_name=cfg.COMET_PROJECT_NAME,
            api_key=os.getenv('COMET_API_KEY'),
            workspace=os.getenv('COMET_WORKSPACE'),
            disabled=cfg.logcomet is False,
        )
        if cfg.logcomet:
            exp_name = cfg_to_exp_name(cfg)
            print(exp_name)
            self.comet_exp.set_name(exp_name)
            self.comet_exp.log_parameters(flatten_json_iterative_solution(cfg))
            self.comet_exp.log_asset(self.logfile)
            self.comet_exp.log_asset_data(json.dumps(cfg, indent=4), file_name='cfg.json')
            self.comet_exp.set_model_graph(str(self.model))
            if cfg.cfg_file:
                self.comet_exp.log_asset(cfg.cfg_file)

        with open(os.path.join(self.path, 'cfg.json'), 'w') as f:
            json.dump(cfg, f, indent=4)


    def print_info(self):
        print('Using config:')
        pprint.pprint(self.cfg)
        print("\n")

        pprint.pprint("Size of train dataset: {}".format(len(self.dataset)))
        # print("\n")
        pprint.pprint("Size of val dataset: {}".format(len(self.dataset_val)))
        print("\n")

        print("Using MAC-Model:")
        pprint.pprint(self.model)
        print("\n")

    def weight_moving_average(self, alpha=0.999):
        for param1, param2 in zip(self.model_ema.parameters(), self.model.parameters()):
            param1.data *= alpha
            param1.data += (1.0 - alpha) * param2.data

    def set_mode(self, mode="train"):
        if mode == "train":
            self.model.train()
            self.model_ema.train()
        else:
            self.model.eval()
            self.model_ema.eval()

    def reduce_lr(self):
        epoch_loss = self.total_epoch_loss # / float(len(self.dataset) // self.batch_size)
        lossDiff = self.prior_epoch_loss - epoch_loss
        if ((lossDiff < 0.015 and self.prior_epoch_loss < 0.5 and self.lr > 0.00002) or \
            (lossDiff < 0.008 and self.prior_epoch_loss < 0.15 and self.lr > 0.00001) or \
            (lossDiff < 0.003 and self.prior_epoch_loss < 0.10 and self.lr > 0.000005)):
            self.lr *= 0.5
            print("Reduced learning rate to {}".format(self.lr))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
        self.prior_epoch_loss = epoch_loss
        self.total_epoch_loss = 0

    def save_models(self, iteration):
        save_model(self.model, self.optimizer, iteration, self.model_dir, model_name="model")
        save_model(self.model_ema, None, iteration, self.model_dir, model_name="model_ema")

    def train_epoch(self, epoch):
        cfg = self.cfg
        total_loss = 0.
        total_correct = 0
        total_samples = 0

        self.labeled_data = iter(self.dataloader)
        self.set_mode("train")

        dataset = tqdm(self.labeled_data, total=len(self.dataloader), ncols=20)

        for data in dataset:
            ######################################################
            # (1) Prepare training data
            ######################################################
            image, question, question_len, answer = data['image'], data['question'], data['question_length'], data['answer']
            answer = answer.long()
            question = Variable(question)
            answer = Variable(answer)

            if cfg.CUDA:
                if self.use_feats == 'spatial':
                    image = image.cuda()
                elif self.use_feats == 'objects':
                    image = [e.cuda() for e in image]
                question = question.cuda()
                answer = answer.cuda().squeeze()
            else:
                question = question
                image = image
                answer = answer.squeeze()

            ############################
            # (2) Train Model
            ############################
            self.optimizer.zero_grad()

            scores = self.model(image, question, question_len)
            loss = self.loss_fn(scores, answer)
            loss.backward()

            if self.cfg.TRAIN.CLIP_GRADS:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.TRAIN.CLIP)

            self.optimizer.step()
            self.weight_moving_average()

            ############################
            # (3) Log Progress
            ############################
            correct = scores.detach().argmax(1) == answer
            total_correct += correct.sum().cpu().item()
            total_loss += loss.item() * answer.size(0)
            total_samples += answer.size(0)

            avg_loss = total_loss / total_samples
            train_accuracy = total_correct / total_samples
            # accuracy = correct.sum().cpu().numpy() / answer.shape[0]

            # if avg_loss == 0:
            #     avg_loss = loss.item()
            #     train_accuracy = accuracy
            # else:
            #     avg_loss = 0.99 * avg_loss + 0.01 * loss.item()
            #     train_accuracy = 0.99 * train_accuracy + 0.01 * accuracy
            # self.total_epoch_loss += loss.item() * answer.size(0)

            dataset.set_description(
                'Epoch: {}; Avg Loss: {:.5f}; Avg Train Acc: {:.5f}'.format(epoch + 1, avg_loss, train_accuracy)
            )

        self.total_epoch_loss = avg_loss

        dict = {
            "loss": avg_loss,
            "accuracy": train_accuracy,
            "avg_loss": avg_loss, # For commet
            "avg_accuracy": train_accuracy, # For commet
        }
        return dict

    def train(self):
        cfg = self.cfg
        print("Start Training")
        for epoch in range(self.start_epoch, self.max_epochs):

            with self.comet_exp.train():
                dict = self.train_epoch(epoch)
                self.reduce_lr()
                dict['epoch'] = epoch + 1
                dict['lr'] = self.lr
                self.comet_exp.log_metrics(dict, epoch=epoch + 1,)

            with self.comet_exp.validate():
                dict = self.log_results(epoch, dict)
                dict['epoch'] = epoch + 1
                dict['lr'] = self.lr
                self.comet_exp.log_metrics(dict, epoch=epoch + 1,)


            if cfg.TRAIN.EALRY_STOPPING:
                if epoch - cfg.TRAIN.PATIENCE == self.previous_best_epoch:
                # if epoch - cfg.TRAIN.PATIENCE == self.previous_best_loss_epoch:
                    print('Early stop')
                    break

        self.comet_exp.log_asset(self.logfile)
        self.save_models(self.max_epochs)
        self.writer.close()
        print("Finished Training")
        print(f"Highest validation accuracy: {self.previous_best_acc} at epoch {self.previous_best_epoch}")

    def log_results(self, epoch, dict, max_eval_samples=None):
        epoch += 1
        self.writer.add_scalar("avg_loss", dict["loss"], epoch)
        self.writer.add_scalar("train_accuracy", dict["accuracy"], epoch)

        metrics = self.calc_accuracy("validation", max_samples=max_eval_samples)
        self.writer.add_scalar("val_accuracy_ema", metrics['acc_ema'], epoch)
        self.writer.add_scalar("val_accuracy", metrics['acc'], epoch)
        self.writer.add_scalar("val_loss_ema", metrics['loss_ema'], epoch)
        self.writer.add_scalar("val_loss", metrics['loss'], epoch)

        print("Epoch: {epoch}\tVal Acc: {acc},\tVal Acc EMA: {acc_ema},\tAvg Loss: {loss},\tAvg Loss EMA: {loss_ema},\tLR: {lr}".
              format(epoch=epoch, lr=self.lr, **metrics))

        if metrics['acc'] > self.previous_best_acc:
            self.previous_best_acc = metrics['acc']
            self.previous_best_epoch = epoch
        if metrics['loss'] < self.previous_best_loss:
            self.previous_best_loss = metrics['loss']
            self.previous_best_loss_epoch = epoch

        if epoch % self.snapshot_interval == 0:
            self.save_models(epoch)

        return metrics

    def calc_accuracy(self, mode="train", max_samples=None):
        self.set_mode("validation")

        if mode == "train":
            loader = self.dataloader
        # elif (mode == "validation") or (mode == 'test'):
        #     loader = self.dataloader_val
        else:
            loader = self.dataloader_val

        total_correct = 0
        total_correct_ema = 0
        total_samples = 0
        total_loss = 0.
        total_loss_ema = 0.
        pbar = tqdm(loader, total=len(loader), desc=mode.upper(), ncols=20)
        for data in pbar:

            image, question, question_len, answer = data['image'], data['question'], data['question_length'], data['answer']
            answer = answer.long()
            question = Variable(question)
            answer = Variable(answer)

            if self.cfg.CUDA:
                if self.use_feats == 'spatial':
                    image = image.cuda()
                elif self.use_feats == 'objects':
                    image = [e.cuda() for e in image]
                question = question.cuda()
                answer = answer.cuda().squeeze()

            with torch.no_grad():
                scores = self.model(image, question, question_len)
                scores_ema = self.model_ema(image, question, question_len)

                loss = self.loss_fn(scores, answer)
                loss_ema = self.loss_fn(scores_ema, answer)


            correct = scores.detach().argmax(1) == answer
            correct_ema = scores_ema.detach().argmax(1) == answer

            total_correct += correct.sum().cpu().item()
            total_correct_ema += correct_ema.sum().cpu().item()

            total_loss += loss.item() * answer.size(0)
            total_loss_ema += loss_ema.item() * answer.size(0)

            total_samples += answer.size(0)

            avg_acc = total_correct / total_samples
            avg_acc_ema = total_correct_ema / total_samples
            avg_loss = total_loss / total_samples
            avg_loss_ema = total_loss_ema / total_samples
            
            pbar.set_postfix({
                'Acc': f'{avg_acc:.5f}',
                'Acc Ema': f'{avg_acc_ema:.5f}',
                'Loss': f'{avg_loss:.5f}',
                'Loss Ema': f'{avg_loss_ema:.5f}',
            })

        return dict(
            acc=avg_acc,
            acc_ema=avg_acc_ema,
            loss=avg_loss,
            loss_ema=avg_loss_ema
        )