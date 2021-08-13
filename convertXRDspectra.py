#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
import torch
import random
import numpy as np
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from torch.utils.data import Dataset, DataLoader
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='./lithium_datasets.pkl',
                        help='path to the input pickle file')
    parser.add_argument('--batch', default=8, type=int,
                        help='batch size of XRD spectra calculation')
    parser.add_argument('--n_aug', default=5, type=int,
                        help='number of data augmentation for peak splitting')
    args = parser.parse_args()
    return args

class XRDspectra():
    def __init__(self, device, thetas=(0,120,0.02), dwfs_range=(0.5,3), sig=0.07):
        self.device = device
        self.thetas = thetas
        self.dwfs = dwfs_range
        self.sig = torch.tensor(sig).to(device)
        
    def pickle_loader(self, pickle_path):
        with open(pickle_path, mode='rb') as f:
            dataset = pickle.load(f)
        print(str(len(dataset[0]))+' data points loaded.')
        return dataset
    
    def labelling(self, attributes):
        materials = [
            ''.join(sorted(attr[1].split(" "))) for attr in attributes
        ]
        unique_materials = sorted(list(set(materials)))
        labels = np.array([unique_materials.index(material) for material in materials])
        print(str(len(unique_materials))+' materials were detected')
        df_materials = pd.DataFrame(labels, index=materials)
        df_materials.to_csv('material_labels.csv')
        return labels, unique_materials
    
    def dataset2tensor(self, dataset):
        X = dataset[0]
        Y = dataset[1]
        Z, _ = self.labelling(dataset[2])
        self.tensors = (X,Y,Z)

    def stochastic_dwf(self, atoms):
        # randomly select the Debye-Waller factors for each atom
        dwfs = []
        for atom in atoms:
            dwfs.append((atom, random.uniform(self.dwfs[0],self.dwfs[1])))
        dwfs = dict(dwfs)
        return dwfs

    def structure2XRDpeaks(self, structure, dwfs, verbose=False):
        xrd = XRDCalculator(
            debye_waller_factors=dwfs,
        ).get_pattern(
            structure,
            scaled=True,
            two_theta_range=(self.thetas[0],self.thetas[1]),
        )
        if verbose:
            plt.vlines(xrd.x, 0, xrd.y, lw=0.5)
        return xrd
    
    def tensors2spectrum(self, xrd):
        xrd_spectrum_x = torch.arange(
            self.thetas[0],
            self.thetas[1],
            self.thetas[2]
        ).to(self.device)
        
        xrd_spectrum_y = torch.zeros(
            int((self.thetas[1] - self.thetas[0]) / self.thetas[2])
        ).to(self.device)
        
        torch.pi = torch.tensor(
            torch.acos(torch.zeros(1)).item() * 2
        ).to(self.device)
        
        xrd_x = torch.from_numpy(xrd.x).to(self.device)
        xrd_y = torch.from_numpy(xrd.y).to(self.device)
        
        for i in range(xrd.y.shape[0]):
            xrd_spectrum_y += xrd_y[i] * (
                1/torch.sqrt(2*torch.pi*self.sig**2)
            ) * torch.exp(
                -(xrd_spectrum_x - xrd_x[i])**2 / (2*self.sig**2)
            )
        xrd_spectrum_y = xrd_spectrum_y / torch.max(xrd_spectrum_y) * 100
        return xrd_spectrum_x, xrd_spectrum_y
    
    def load(self, pickle_path):
        dataset = self.pickle_loader(pickle_path)
        self.dataset2tensor(dataset)
    
    def forward(self, index):
        structure = self.tensors[0][index]
        atom = self.tensors[1][index]
        dwf = self.stochastic_dwf(atom)
        xrd = self.structure2XRDpeaks(structure,dwf)
        _, xrd_spectrum = self.tensors2spectrum(xrd)
        label = torch.tensor(self.tensors[2][index]).to(self.device)
        return xrd_spectrum, label

class CustomDataset(Dataset):
    def __init__(self, pickle_path):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.xrd = XRDspectra(device)
        self.xrd.load(pickle_path)
        
    def __getitem__(self, index):
        X, Y = self.xrd.forward(index)
        return X, Y
    
    def __len__(self):
        return len(self.xrd.tensors[0])
        
def create_data_loader(pickle_path, batch_size, shuffle=False):
    ds = CustomDataset(pickle_path)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return dataloader

if __name__ == '__main__':
    args = parse_args()
    dataloader = create_data_loader(
        args.input,
        args.batch,
        shuffle=False,
    )

    inputs = []
    labels = []
    for epoch in range(args.n_aug):
        print('>>>> '+str(epoch+1)+' / '+str(args.n_aug) + ' epoch')
        for batch_idx, (input, label) in enumerate(dataloader):
            print(str(batch_idx+1)+' / '+str(len(dataloader))+' converting')
            inputs.extend(input.cpu().numpy().copy())
            labels.extend(label.cpu().numpy().copy())

        xrd_datasets = (
            np.array(inputs),
            np.array(labels),
        )
        with open("XRD_epoch"+str(epoch+1)+".pkl", mode="wb") as f:
            pickle.dump(xrd_datasets, f)
        print('epoch'+str(epoch+1)+' saved')
    print('Successfully converted')
