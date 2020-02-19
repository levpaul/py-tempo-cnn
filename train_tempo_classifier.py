#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse
from torch.utils.data import Dataset, DataLoader

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # TODO: Add network layers here

    def forward(self, x):
        x = x # TODO: Add actual network links here

class FMASpectrogramsDataset(Dataset):
    def __init__(self, train):
        if train:
            self.data = np.load('data/fma-train-data.npy')
            self.labels = np.load('data/fma-train-labels.npy')
        else:
            self.data = np.load('data/fma-test-data.npy')
            self.labels = np.load('data/fma-test-labels.npy')

    def __len__(self):
        return self.labels.size

    def __getitem__(self, idx):
        return (self.data[idx], self.labels[idx])


def parse_args():
    pa = argparse.ArgumentParser()
    pa.add_argument('-i', '--isolated', required=False)
    return pa.parse_args()


def run():
    args = parse_args()
    print('TODO: implement the rest')

if __name__ == '__main__':
    run()
