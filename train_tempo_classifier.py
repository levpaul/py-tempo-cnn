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
        self.train = train
        # TODO: return np arrays depending on train or test

    def __len__(self):
        return 0 # TODO: impl

    def __getitem__(self, idx):
        return 0 # TODO: return spectrogram + label


def parse_args():
    pa = argparse.ArgumentParser()
    pa.add_argument('-i', '--isolated', required=False)
    return pa.parse_args()


def run():
    args = parse_args()
    print('TODO: implement the rest')

if __name__ == '__main__':
    run()
