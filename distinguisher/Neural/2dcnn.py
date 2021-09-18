import os
import time
import numpy as np
from PIL import Image
# import progressbar
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms

import utils
import models

########################################
# utils.py                            #
########################################

class Record(object):
    def __init__(self):
        self.loss = 0
        self.count = 0

    def add(self, value):
        self.loss += value
        self.count += 1

    def mean(self):
        return self.loss / self.cou

def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def accuracy(scores, targets, k=1):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, dim=1, largest=True, sorted=True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() / batch_size

########################################
# models.py                            #
########################################

class trace_encoder_32(nn.Module):
    def __init__(self, dim, nc=1):
        super(trace_encoder_32, self).__init__()
        self.dim = dim
        # nf = 64
        nf = 8
        # state size. (nc) x 32 x 32
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 16 x 16
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 8 x 8
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 4 x 4
        self.c4 = nn.Sequential(
                nn.Conv2d(nf * 4, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )
        self.apply(weights_init)

    def forward(self, x):
        x = F.normalize(x)
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        return h4.view(-1, self.dim)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class OramDataset(Dataset):
    def __init__(self, args, split):
        self.args = args
        super(OramDataset).__init__()
        file_path = args.input_root + ('train.txt' if split == 'train' else 'val.txt')
        with open(file_path, 'r') as f:
            seq_list = f.readlines()

        self.side_list = []
        self.label_list = []
        self.label_set = set()

        for seq in tqdm(seq_list):
            side, label = self.preprocess(seq.strip().split(' '))
            self.side_list.append(side)
            self.label_list.append(label)
        assert len(self.side_list) == len(self.label_list)
        print(list(self.label_set))

    def preprocess(self, seq):
        label = 0 if seq[0] == '__label__1' else 1
        self.label_set.add(seq[0])
        data_list = seq[1:]
        assert len(data_list) % 2 == 0
        loc_list = []
        for i in range(len(data_list) // 2):
            op = data_list[i * 2]
            loc = data_list[i * 2 + 1]
            if op == 'read':
                loc_list.append(int(loc))
            elif op == 'write':
                loc_list.append(-1 * int(loc))
            elif op == 'allocate':
                continue
            else:
                raise NotImplementedError
        pad_length = self.args.trace_w * self.args.trace_w * self.args.trace_c
        if len(loc_list) < pad_length:
            loc_list += [0] * (pad_length - len(loc_list))
        else:
            loc_list = loc_list[:pad_length]

        return torch.from_numpy(np.array(loc_list).astype(np.float32)), torch.LongTensor([label]).squeeze()

    def __len__(self):
        return len(self.side_list)

    def __getitem__(self, index):
        side = self.side_list[index]
        side = side.view([self.args.trace_c, self.args.trace_w, self.args.trace_w])
        label = self.label_list[index]
        return side, label


class Engine(object):
    def __init__(self, args):
        self.args = args
        self.epoch = 0
        self.mse = nn.MSELoss().cuda()
        self.l1 = nn.L1Loss().cuda()
        self.bce = nn.BCELoss().cuda()
        self.ce = nn.CrossEntropyLoss().cuda()
        self.real_label = 1
        self.fake_label = 0
        self.init_model_optimizer()

    def init_model_optimizer(self):
        print('Initializing Model and Optimizer...')
        # self.enc = models.__dict__['SelfAttenTraceEnc'](trace_dim=2,
        #                                                 out_dim=self.args.nz, 
        #                                                 x_len=256,
        #                                                 y_len=256,
        #                                                 d_k=8,
        #                                                 d_v=8)
        self.enc = models.__dict__['trace_encoder_%d' % self.args.trace_w](dim=self.args.nz, nc=args.trace_c)
        print('enc params: ', count_parameters(self.enc))
        #self.enc = models.__dict__['encoder_%d' % self.args.image_size](dim=self.args.nz, nc=self.args.nc)
        self.enc = torch.nn.DataParallel(self.enc).cuda()

        self.fc = nn.Sequential(
                nn.Linear(self.args.nz, 2)
            )
        print('fc params: ', count_parameters(self.fc))
        self.fc = torch.nn.DataParallel(self.fc).cuda()

        self.optim = torch.optim.Adam(
                        list(self.enc.module.parameters()) + \
                        list(self.fc.module.parameters()),
                        lr=self.args.lr,
                        betas=(self.args.beta1, 0.999)
                        )

    def save_model(self, path):
        print('Saving Model on %s ...' % (path))
        state = {
            'enc': self.enc.module.state_dict(),
            'fc': self.fc.module.state_dict()
        }
        torch.save(state, path)

    def load_model(self, path):
        print('Loading Model from %s ...' % (path))
        ckpt = torch.load(path)
        self.enc.module.load_state_dict(ckpt['enc'])
        self.fc.module.load_state_dict(ckpt['fc'])

    def zero_grad(self):
        self.enc.zero_grad()
        self.fc.zero_grad()

    def set_train(self):
        self.enc.train()
        self.fc.train()

    def set_eval(self):
        self.enc.eval()
        self.fc.eval()

    def train(self, data_loader):
        with torch.autograd.set_detect_anomaly(True):
            self.epoch += 1
            self.set_train()
            record = utils.Record()
            record_acc = utils.Record()
            start_time = time.time()
            for i, (trace, label) in enumerate(tqdm(data_loader)):
                label = label.cuda()
                trace = trace.cuda()

                self.zero_grad()
                encoded = self.enc(trace)
                pred = self.fc(encoded)
                
                #recons_err = self.mse(decoded, image)
                loss = self.ce(pred, label)
                loss.backward()
                self.optim.step()

                record.add(loss.item())
                record_acc.add(utils.accuracy(pred, label))

            print('----------------------------------------')
            print('Epoch: %d' % self.epoch)
            print('Costs Time: %.2f s' % (time.time() - start_time))
            print('Loss: %f' % (record.mean()))
            print('Acc: %f' % (record_acc.mean()))

            
    def test(self, data_loader):
        #with torch.autograd.set_detect_anomaly(True):
        self.set_eval()
        record = utils.Record()
        record_acc = utils.Record()
        start_time = time.time()
        with torch.no_grad():
            for i, (trace, label) in enumerate(tqdm(data_loader)):
                label = label.cuda()
                trace = trace.cuda()

                encoded = self.enc(trace)
                pred = self.fc(encoded)
                
                loss = self.ce(pred, label)
                record.add(loss.item())
                record_acc.add(utils.accuracy(pred, label))

            print('----------------------------------------')
            print('Test.')
            print('Costs Time: %.2f s' % (time.time() - start_time))
            print('Loss: %f' % (record.mean()))
            print('Acc: %f' % (record_acc.mean()))


if __name__ == '__main__':
    import argparse
    import random

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    import utils
    from data_loader import DataLoader

    parser = argparse.ArgumentParser()

    #parser.add_argument('--exp_name', type=str, default=('final_pretrain_celeba_%s' % side))
    parser.add_argument('--exp_name', type=str, default='dis_ex2') # dis_exp 002.pth 57.28

    # parser.add_argument('--input_root', type=str, default='/home/user/oram/temp/extremely_lengthy/')
    parser.add_argument('--input_root', type=str, default='/home/user/oram/temp/48/')

    parser.add_argument('--output_root', type=str, default='/export/d2/root/dataset/output/oram/')
    parser.add_argument('--trace_w', type=int, default=32)
    parser.add_argument('--trace_c', type=int, default=7)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--D_lr', type=float, default=2e-3)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--nz', type=int, default=32)
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--num_epoch', type=int, default=3)
    parser.add_argument('--test_freq', type=int, default=1)
    # parser.add_argument('--n_class', type=int, default=-1)

    args = parser.parse_args()

    print(args.exp_name)

    manual_seed = random.randint(1, 10000)
    #manual_seed = 202
    print('Manual Seed: %d' % manual_seed)

    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    utils.make_path(args.output_root)
    utils.make_path(args.output_root + args.exp_name)

    args.ckpt_root = args.output_root + args.exp_name + '/ckpt/'

    utils.make_path(args.ckpt_root)

    with open(args.output_root + args.exp_name + '/args.json', 'w') as f:
        json.dump(args.__dict__, f)

    loader = DataLoader(args)

    train_dataset = OramDataset(args, split='train')
    test_dataset = OramDataset(args, split='test')

    model = Engine(args)

    train_loader = loader.get_loader(train_dataset)
    test_loader = loader.get_loader(test_dataset)

    # model.load_model((args.ckpt_root + '026.pth'))
    # model.test(test_loader)
    
    for i in range(model.epoch, args.num_epoch):
        model.train(train_loader)
        if (i + 0) % args.test_freq == 0:
            model.test(test_loader)
            model.save_model((args.ckpt_root + '%03d.pth') % (i + 1))
    model.save_model((args.ckpt_root + 'final.pth'))
