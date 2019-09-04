from __future__ import print_function

import sys
import os
import shutil
from six.moves import range
import pprint
from tqdm import tqdm

from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import mkdir_p, save_model, load_vocab
from datasets import ClevrDataset, collate_fn
import mac
from radam import RAdam


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
            sys.stdout = Logger(logfile=os.path.join(self.path, "logfile.log"))

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

        # load dataset
        cogent = cfg.DATASET.COGENT
        if cogent:
            print(f'Using CoGenT {cogent.upper()}')
        sample = cfg.SAMPLE
        if cfg.TRAIN.FLAG:
            self.dataset = ClevrDataset(data_dir=self.data_dir, split="train" + cogent, sample=sample)
            self.dataloader = DataLoader(dataset=self.dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                                        num_workers=cfg.WORKERS, drop_last=True, collate_fn=collate_fn)
        else:
            self.dataset = []
            self.dataloader = []

        if cfg.TRAIN.FLAG or cfg.EVAL:
            self.dataset_val = ClevrDataset(data_dir=self.data_dir, split="val" + cogent, sample=sample)
            self.dataloader_val = DataLoader(dataset=self.dataset_val, batch_size=cfg.TEST_BATCH_SIZE, drop_last=False,
                                            shuffle=False, num_workers=cfg.WORKERS, collate_fn=collate_fn)

        elif cfg.TEST:
            self.dataset_val = ClevrDataset(data_dir=self.data_dir, split="test" + cogent, sample=sample)
            self.dataloader_val = DataLoader(dataset=self.dataset_val, batch_size=cfg.TEST_BATCH_SIZE, drop_last=False,
                                            shuffle=False, num_workers=cfg.WORKERS, collate_fn=collate_fn)

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

        self.total_epoch_loss = 0
        self.prior_epoch_loss = 10

        self.print_info()
        self.loss_fn = torch.nn.CrossEntropyLoss().cuda()

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
        total_loss = 0
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
                image = image.cuda()
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
            "avg_loss": avg_loss,
            "train_accuracy": train_accuracy
        }
        return dict

    def train(self):
        cfg = self.cfg
        print("Start Training")
        for epoch in range(self.start_epoch, self.max_epochs):
            dict = self.train_epoch(epoch)
            self.reduce_lr()
            self.log_results(epoch, dict)
            if cfg.TRAIN.EALRY_STOPPING:
                if epoch - cfg.TRAIN.PATIENCE == self.previous_best_epoch:
                    break

        self.save_models(self.max_epochs)
        self.writer.close()
        print("Finished Training")
        print(f"Highest validation accuracy: {self.previous_best_acc} at epoch {self.previous_best_epoch}")

    def log_results(self, epoch, dict, max_eval_samples=None):
        epoch += 1
        self.writer.add_scalar("avg_loss", dict["avg_loss"], epoch)
        self.writer.add_scalar("train_accuracy", dict["train_accuracy"], epoch)

        val_accuracy, val_accuracy_ema = self.calc_accuracy("validation", max_samples=max_eval_samples)
        self.writer.add_scalar("val_accuracy_ema", val_accuracy_ema, epoch)
        self.writer.add_scalar("val_accuracy", val_accuracy, epoch)

        print("Epoch: {}\tVal Acc: {},\tVal Acc EMA: {},\tAvg Loss: {},\tLR: {}".
              format(epoch, val_accuracy, val_accuracy_ema, dict["avg_loss"], self.lr))

        if val_accuracy > self.previous_best_acc:
            self.previous_best_acc = val_accuracy
            self.previous_best_epoch = epoch

        if epoch % self.snapshot_interval == 0:
            self.save_models(epoch)

    def calc_accuracy(self, mode="train", max_samples=None):
        self.set_mode("validation")

        if mode == "train":
            loader = self.dataloader
            # num_imgs = len(self.dataset)
        elif (mode == "validation") or (mode == 'test'):
            loader = self.dataloader_val
            # num_imgs = len(self.dataset_val)

        # batch_size = 256
        # total_iters = num_imgs // batch_size
        # if max_samples is not None:
        #     max_iter = max_samples // batch_size
        # else:
        #     max_iter = None

        # all_accuracies = []
        total_correct = 0
        total_correct_ema = 0
        total_samples = 0
        # all_accuracies_ema = []
        pbar = tqdm(loader, total=len(loader), desc=mode.upper(), ncols=20)
        for data in pbar:
            # try:
            #     data = next(eval_data)
            # except StopIteration:
            #     break
            # if max_iter is not None and _iteration == max_iter:
            #     break

            image, question, question_len, answer = data['image'], data['question'], data['question_length'], data['answer']
            answer = answer.long()
            question = Variable(question)
            answer = Variable(answer)

            if self.cfg.CUDA:
                image = image.cuda()
                question = question.cuda()
                answer = answer.cuda().squeeze()

            with torch.no_grad():
                scores = self.model(image, question, question_len)
                scores_ema = self.model_ema(image, question, question_len)

            correct_ema = scores_ema.detach().argmax(1) == answer
            total_correct_ema += correct_ema.sum().cpu().item()
            # accuracy_ema = correct_ema.sum().cpu().numpy() / answer.shape[0]

            # all_accuracies_ema.append(accuracy_ema)

            correct = scores.detach().argmax(1) == answer
            total_correct += correct.sum().cpu().item()
            # accuracy = correct.sum().cpu().numpy() / answer.shape[0]
            # all_accuracies.append(accuracy)
            total_samples += answer.size(0)

            # pbar.set_description(
            #     'Avg Acc: {:.5f}; Avg Acc: {:.5f}'.format(total_correct / total_samples, total_correct_ema / total_samples)
            # )
            pbar.set_postfix({
                'Acc': f'{total_correct / total_samples:.5f}',
                'Ema Acc': f'{total_correct_ema / total_samples:.5f}',
            })

        accuracy_ema = total_correct_ema / total_samples
        accuracy = total_correct / total_samples

        return accuracy, accuracy_ema
