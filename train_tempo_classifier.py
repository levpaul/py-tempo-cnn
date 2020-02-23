#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse
import random
from torch.utils.data import Dataset, DataLoader

epochs=10000
bsize=100

spectro_window_size = 256

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Spectros come in at 40 x 256
        self.conv1 = nn.Conv2d(1, 6, 3) # 6x38x254
        self.conv2 = nn.Conv2d(6, 16, 3) # 16x36x252 -> Maxpool.2= 16x18x126
        self.conv3 = nn.Conv2d(16, 1, 5) # 16x36x252 -> Maxpool.2= 16x18x126
        self.fc1 = nn.Linear(14*122, 50) # 120
        # self.fc2 = nn.Linear(120, 50) # 50
        self.fc3 = nn.Linear(50, 256) # 256 final out

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x,2)
        x = F.dropout2d(x, 0.1)
        x = self.conv3(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.fc2(x)
        # x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output

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
        d = self.data[idx]
        max_idx = d.shape[-1] - spectro_window_size
        lower_idx = random.randint(0, max_idx)
        upper_idx = lower_idx + spectro_window_size
        return (self.data[idx,:,:,lower_idx:upper_idx], self.labels[idx].astype(np.long))

def train(net, loader, device, opt):
    net.train()
    running_loss = 0.
    for bidx, (data, targets) in enumerate(loader):
        data, targets = data.to(device), targets.to(device)
        opt.zero_grad()

        outs = net(data)
        # loss = loss_func(outs, targets)
        loss = F.nll_loss(outs, targets)
        loss.backward() # <- compute backprop derivs
        opt.step()

        running_loss += loss.item()
        if bidx % 50 == 49:
            print('[{:5d}] loss: {:0.3f}'.format(bidx+1, running_loss/50))
            running_loss = 0.

def test(net, loader, device):
    net.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            out = net(data)
            loss += F.nll_loss(out, targets, reduction='sum').item()
            pred = out.argmax(dim=1, keepdim=True)
            correct += pred.eq(targets.view_as(pred)).sum().item()

    loss /= len(loader.dataset)
    print("Test set: Average loss: {:.4f}, Acc: {}/{} ({:.0f}%)\n".format(
        loss, correct, len(loader.dataset),
        100. * correct / len(loader.dataset)
    ))

def parse_args():
    pa = argparse.ArgumentParser()
    pa.add_argument('-i', '--isolated', required=False)
    return pa.parse_args()

def run():
    args = parse_args()

    print('Initializing Data Loaders...')
    tr_loader = DataLoader(FMASpectrogramsDataset(train=True),
                           batch_size=bsize, shuffle=True, num_workers=1, pin_memory=True)
    te_loader = DataLoader(FMASpectrogramsDataset(train=False),
                           batch_size=bsize, shuffle=True, num_workers=1, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Initializing network and optimizer')
    net = Net().cuda().half()
    opt = optim.SGD(net.parameters(), lr=0.01)

    for e in range(1,epochs+1):
        print('Begining training at epoch {:d}'.format(e))
        train(net, tr_loader, device, opt)
        test(net, tr_loader, device)
        torch.save(net.state_dict(), 'saves/nn-ep-{:03d}.pt'.format(e))

    # Load like so:
    # newnet = Net()
    # newnet.load_state_dict(torch.load('nn.pt'))
    print('TODO: implement the rest')

if __name__ == '__main__':
    run()
