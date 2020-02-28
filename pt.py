#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

bsize = 128
loader_workers = 1
num_epochs = 5

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3) # 6x26x26
        self.conv2 = nn.Conv2d(6, 16, 3) # 16x24x24 -> Maxpool.2= 16x12x12
        self.fc1 = nn.Linear(16*12*12, 50) # 120
        # self.fc2 = nn.Linear(120, 50) # 50
        self.fc3 = nn.Linear(50, 10) # 10 final out

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x,2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.fc2(x)
        # x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output

class MyMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, train):
        tfmnist = tf.keras.datasets.mnist
        (xtr, ytr), (xte, yte) = tfmnist.load_data()

        if train:
            xs = xtr.shape
            self.xdata = xtr.reshape(xs[0], 1, xs[1], xs[2]).astype(np.float32)
            self.ydata = ytr.astype(np.int64)
        else:
            xs = xte.shape
            self.xdata = xte.reshape(xs[0], 1, xs[1], xs[2]).astype(np.float32)
            self.ydata = yte.astype(np.int64)

    def __len__(self):
        return self.xdata.shape[0]

    def __getitem__(self, idx):
        return self.xdata[idx], self.ydata[idx]


def train(net, opt, loader, device):
    net.train()
    loss_func = nn.MSELoss()
    running_loss = 0.0
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

def run():
    train_loader = torch.utils.data.DataLoader(MyMNISTDataset(train=True),
        batch_size=bsize, shuffle=True, num_workers=loader_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(MyMNISTDataset(train=False),
                                           batch_size=bsize, shuffle=True, num_workers=loader_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)
    opt = optim.SGD(net.parameters(), lr=0.01)
    for e in range(1, num_epochs+1):
        print('Starting epoch {:d}'.format(e))
        train(net, opt, train_loader, device)
        test(net, test_loader, device)
    print("Done")

if __name__ == '__main__':
    run()
