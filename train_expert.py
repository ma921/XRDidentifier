#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter
from torch.utils.data import Dataset, DataLoader
import math
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from net1d import *
import argparse
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='./XRD_epoch5.pkl',
                        help='path to the input pickle file')
    parser.add_argument('--output', default='learning_curve.csv',
                        help='save learning curve as csv file')
    parser.add_argument('--batch', default=16, type=int,
                        help='batch size')
    parser.add_argument('--n_epoch', default=100, type=int,
                        help='number of training iteration')
    args = parser.parse_args()
    return args

# copied from https://github.com/4uiiurz1/pytorch-adacos/blob/master/metrics.py
class AdaCos(nn.Module):
    def __init__(self, num_features, num_classes, m=0.50):
        super(AdaCos, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = math.sqrt(2) * math.log(num_classes - 1)
        self.m = m
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # feature re-scale
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / input.size(0)
            # print(B_avg)
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
        output = self.s * logits

        return output

# copied and modified from https://github.com/cvqluu/Angular-Penalty-Softmax-Losses-Pytorch
class AngularPenaltySMLoss(nn.Module):
    def __init__(self, loss_type='cosface', eps=1e-7, s=None, m=None):
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in  ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.eps = eps

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        wf = x
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)) + self.m)
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)))

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)

def spectra_loader(pickle_path):
    with open(pickle_path, mode="rb") as f:
        xrd_datasets = pickle.load(f)
    return xrd_datasets

def normalise(spectra):
    if type(spectra) is np.ndarray:
        max_I = np.max(spectra)
        min_I = np.min(spectra)    
    elif type(spectra) is torch.Tensor:
        max_I = max(spectra)
        min_I = min(spectra)
    spectra_normed = (spectra - min_I) / (max_I - min_I)
    return spectra_normed

def random_data_split(spectra, labels, settings):
    thresh1 = round(settings[0]*settings[5])
    thresh2 = round((settings[0] - thresh1)/2 + thresh1)
    l = list(range(settings[0]))
    lr = random.sample(l, settings[0])
    
    data_train = np.array([spectra[idx] for idx in lr[:thresh1]])
    data_val = np.array([spectra[idx] for idx in lr[thresh1:thresh2]])
    data_test = np.array([spectra[idx] for idx in lr[thresh2:]])
    labels_train = np.array([labels[idx] for idx in lr[:thresh1]])
    labels_val = np.array([labels[idx] for idx in lr[thresh1:thresh2]])
    labels_test = np.array([labels[idx] for idx in lr[thresh2:]])
    return (data_train, data_val, data_test), (labels_train, labels_val, labels_test)

# copied part of code from https://github.com/PV-Lab/autoXRD
class data_augmentation():
    def __init__(self, settings, settings_aug):
        self.settings = settings
        self.settings_aug = settings_aug
    
    def peak_elimination(self, xrd):
        random_window = torch.from_numpy(
            np.random.choice([0,0,1], self.settings_aug[0]),
        ).to(self.settings[4])
        dum1 = random_window.repeat(self.settings[2]//self.settings_aug[0])
        xrd_el = torch.mul(xrd, dum1)
        return xrd_el
    
    def peak_scaling(self, xrd):
        random_window = torch.rand(self.settings_aug[0]).to(self.settings[4])
        dum2 = random_window.repeat(self.settings[2]//self.settings_aug[0])
        xrd_sc = torch.mul(xrd, dum2)
        return xrd_sc
    
    def peak_shift(self, xrd):
        cut = torch.randint(
            -self.settings_aug[1],
            self.settings_aug[1],
            (1,),
        ).to(self.settings[4])
        if cut >= 0:
            xrd_sh = torch.cat(
                [xrd[cut:], torch.zeros([cut,]).to(self.settings[4])],
                0,
            )
        else:
            xrd_sh = torch.cat(
                [
                    xrd[0:self.settings[2]+cut,],
                    torch.zeros([-cut,]).to(self.settings[4])
                ],
                0,
            )
        return xrd_sh
    
    def forward(self, xrd):
        if torch.rand(1) < self.settings_aug[2]:
            xrd = self.peak_elimination(xrd)
        if torch.rand(1) < self.settings_aug[3]:
            xrd = self.peak_scaling(xrd)
        if torch.rand(1) < self.settings_aug[4]:
            xrd = self.peak_shift(xrd)
        return normalise(xrd)
    
class AugmentedDataset(Dataset):
    def __init__(self, tensors, settings, settings_aug):
        self.tensors = tensors
        self.settings = settings
        self.settings_aug = settings_aug
        self.augmentation = data_augmentation(settings, settings_aug)
    
    def __getitem__(self, index):
        x = torch.from_numpy(self.tensors[0][index][0]).to(self.settings[4])
        x = self.augmentation.forward(x).unsqueeze(0)
        y = torch.tensor(
            self.tensors[1][index].astype(np.float32)
        ).to(self.settings[4])
        return x, y
    
    def __len__(self):
        return len(self.tensors[0])
    
def AugmentedDataloader(spectra, labels, settings, settings_aug):
    tensors = (spectra, labels)
    ds = AugmentedDataset(
        tensors,
        settings,
        settings_aug,
    )
    loader = DataLoader(
        ds,
        batch_size=settings[6],
        shuffle=True,
    )
    return loader

def dataloader_preparation(pickle_path, split_ratio=0.7, batch_size=8):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # load dataset
    xrd_datasets = spectra_loader(pickle_path)
    spectra = normalise(xrd_datasets[0][:, np.newaxis, :])
    labels = xrd_datasets[1]

    # measure the numbers of dataset shape
    n_samples, n_channel, n_length = spectra.shape
    n_class = len(np.unique(labels))
    settings = (n_samples, n_channel, n_length, n_class, device, split_ratio, batch_size)
    settings_aug = (100, 120, 0.2, 0.2, 0.5)
    # (window size, max peak shift size, probability of peak elimination,
    #  probability of peak scailing, probability of peak shift)
    
    # dataloaders
    spectra_split, labels_split = random_data_split(spectra, labels, settings)
    dataloader_train = AugmentedDataloader(
        spectra_split[0],
        labels_split[0],
        settings,
        settings_aug,
    )

    dataloader_val = DataLoader(
        MyDataset(spectra_split[1],labels_split[1]),
        batch_size=settings[6],
    )

    dataloader_test = DataLoader(
        MyDataset(spectra_split[2],labels_split[2]),
        batch_size=settings[6],
    )
    
    # compile dataloaders and settings
    dataloaders = (dataloader_train, dataloader_val, dataloader_test)
    return dataloaders, settings

def load_model(settings):
    model = Net1D(
        in_channels=settings[1],
        base_filters=64,
        ratio=1.0,
        filter_list=[64,160,160,400,400,1024,1024],
        m_blocks_list=[2,2,2,3,3,4,4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        n_classes=settings[3],
        verbose=False,
    )
    model.dense = AdaCos(1024,settings[3])
    model.to(settings[4])
    return model

# copied from https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
def top_k(pred, label, k:int = 1):
    labels_dim = 1
    k_labels = torch.topk(input=pred, k=k, dim=1, largest=True, sorted=True)[1]
    a = ~torch.prod(
        input = torch.abs(label.unsqueeze(labels_dim) - k_labels),
        dim=labels_dim,
    ).to(torch.bool)
    a = a.to(torch.int8)
    y_pred = a * label + (1-a) * k_labels[:,0]
    acc = accuracy_score(y_pred, label)*100
    return acc

def record_learning_curve(lc_name, epoch, results, loss_train, acc_train, loss_val, acc_val):
    results[epoch, :] = np.array([loss_train, acc_train, loss_val, acc_val])
    df = pd.DataFrame(results, columns=['loss_train', 'acc_train', 'loss_val', 'acc_val'])
    df.to_csv(lc_name)

def save_model(best_acc, epoch, model):
    print('--------> The best model has been replaced.')
    print('epoch: '+str(epoch)+' | best_acc: '+str(best_acc))
    model_path = './regnet1d_adacos_epoch'+str(epoch)+'.pt'
    torch.save(model.state_dict(), model_path)
    print('The best model has been saved in '+model_path) 

def train(dataloaders, settings, model, criterion, optimizer):
    running_loss = 0.0
    running_corrects = 0.0
    model.train()
    model.zero_grad()
    
    for batch_idx, batch in enumerate(dataloaders[0]):
        # train
        input, label = tuple(t.to(settings[4]) for t in batch)
        label = label.long()
        pred = model(input)
        loss = criterion(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # evaluate
        running_corrects += top_k(pred, label, k=5) * len(label)
        running_loss += loss.item()
        print('[Train] batch: '+str(batch_idx+1)+' | loss: '+str(loss.item()))
    
    # summarise
    n_train = round(settings[0] * settings[5])
    epoch_loss = running_loss / n_train
    epoch_acc = running_corrects / n_train
    print('[Train total] loss: '+str(epoch_loss)+' | acc: '+str(epoch_acc))
    return epoch_loss, epoch_acc

def val(dataloader, settings, model, criterion):
    running_loss = 0.0
    running_corrects = 0.0
    model.eval()
    model.zero_grad()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # test
            input, label = tuple(t.to(settings[4]) for t in batch)
            label = label.long()
            pred = model(input)
            loss = criterion(pred, label)

            # evaluate
            running_corrects += top_k(pred, label, k=5) * len(label)
            running_loss += loss.item()
            print('[Val] batch: '+str(batch_idx+1)+' | loss: '+str(loss.item()))
    
    # summarise
    n_test = round(settings[0] * (1 - settings[5])/2)
    epoch_loss = running_loss / n_test
    epoch_acc = running_corrects / n_test
    print('[Val total] loss: '+str(epoch_loss)+' | acc: '+str(epoch_acc))
    return epoch_loss, epoch_acc

if __name__ == '__main__':
    args = parse_args()
    dataloaders, settings = dataloader_preparation(
        args.input,
        batch_size=args.batch,
    )
    model = load_model(settings)
    criterion = AngularPenaltySMLoss(loss_type='cosface').to(settings[4])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_acc = 0.0
    results = np.zeros([args.n_epoch, 4])
    for epoch in range(args.n_epoch):
        print('>>>>>> epoch '+str(epoch)+' starts')
        loss_train, acc_train = train(dataloaders, settings, model, criterion, optimizer)
        loss_val, acc_val = val(dataloaders[1], settings, model, criterion)
        record_learning_curve(
            args.output,
            epoch,
            results,
            loss_train,
            acc_train,
            loss_val,
            acc_val,
        )

        # save better model
        if best_acc <= acc_val:
            best_acc = acc_val
            save_model(best_acc, epoch, model)
            loss_test, acc_test = val(dataloaders[2], settings, model, criterion)
