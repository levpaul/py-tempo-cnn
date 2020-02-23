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

class PaperNet(nn.Module):
    def __init__(self):
        super(PaperNet, self).__init__()
        # Short Conv NN x3

        # Spectros come in at 40 x 256
        # ====== SHORT FILTERS ======
        self.sf_conv1 = nn.Conv2d(1, 16, (1,5)) # 16x36x252
        self.sf_conv1_bn = nn.BatchNorm2d(1)
        self.sf_conv2 = nn.Conv2d(16, 16, (1,5)) # 16x32x248
        self.sf_conv2_bn = nn.BatchNorm2d(16)
        self.sf_conv3 = nn.Conv2d(16, 16, (1,5)) # 16x28x244
        self.sf_conv3_bn = nn.BatchNorm2d(16)

        # Multi filter NN x4
        # ====== MULTI FILTER MODS ======
        m=5
        input_chans=16
        self.mf1_ap = nn.AvgPool2d((m,1))
        self.mf1_bn = nn.BatchNorm2d(input_chans)
        self.mf1_conv1 = nn.Conv2d(input_chans,24,(1,32))
        self.mf1_conv2 = nn.Conv2d(input_chans,24,(1,64))
        self.mf1_conv3 = nn.Conv2d(input_chans,24,(1,96))
        self.mf1_conv4 = nn.Conv2d(input_chans,24,(1,128))
        self.mf1_conv5 = nn.Conv2d(input_chans,24,(1,192))
        self.mf1_conv6 = nn.Conv2d(input_chans,24,(1,244)) # This is a single fullscale, cutoff by previous CNNs
        self.mf1_conv_final = nn.Conv2d(24,36,1)

        m=2
        input_chans=36
        self.mf2_ap = nn.AvgPool2d((m,1))
        self.mf2_bn = nn.BatchNorm2d(input_chans)
        self.mf2_conv1 = nn.Conv2d(input_chans,24,(1,32))
        self.mf2_conv2 = nn.Conv2d(input_chans,24,(1,64))
        self.mf2_conv3 = nn.Conv2d(input_chans,24,(1,96))
        self.mf2_conv4 = nn.Conv2d(input_chans,24,(1,128))
        self.mf2_conv5 = nn.Conv2d(input_chans,24,(1,192))
        self.mf2_conv6 = nn.Conv2d(input_chans,24,(1,244)) # This is a single fullscale, cutoff by previous CNNs
        self.mf2_conv_final = nn.Conv2d(24,36,1)

        self.mf3_ap = nn.AvgPool2d((m,1))
        self.mf3_bn = nn.BatchNorm2d(input_chans)
        self.mf3_conv1 = nn.Conv2d(input_chans,24,(1,32))
        self.mf3_conv2 = nn.Conv2d(input_chans,24,(1,64))
        self.mf3_conv3 = nn.Conv2d(input_chans,24,(1,96))
        self.mf3_conv4 = nn.Conv2d(input_chans,24,(1,128))
        self.mf3_conv5 = nn.Conv2d(input_chans,24,(1,192))
        self.mf3_conv6 = nn.Conv2d(input_chans,24,(1,244)) # This is a single fullscale, cutoff by previous CNNs
        self.mf3_conv_final = nn.Conv2d(24,36,1)

        # GPU NOT ENOUGH RAM
        # self.mf4_ap = nn.AvgPool2d((m,1))
        # self.mf4_bn = nn.BatchNorm2d(input_chans)
        # self.mf4_conv1 = nn.Conv2d(input_chans,24,(1,32))
        # self.mf4_conv2 = nn.Conv2d(input_chans,24,(1,64))
        # self.mf4_conv3 = nn.Conv2d(input_chans,24,(1,96))
        # self.mf4_conv4 = nn.Conv2d(input_chans,24,(1,128))
        # self.mf4_conv5 = nn.Conv2d(input_chans,24,(1,192))
        # self.mf4_conv6 = nn.Conv2d(input_chans,24,(1,244)) # This is a single fullscale, cutoff by previous CNNs
        # self.mf4_conv_final = nn.Conv2d(24,36,1)

        # ====== DENSE LAYERS ======
        # Dense Layers
        # mfmod_size = 36*40908
        mfmod_size = 205632
        self.dl_bn1 = nn.BatchNorm1d(mfmod_size)
        self.dl_do = nn.Dropout(0.5)
        self.dl_fc1 = nn.Linear(mfmod_size, 64)
        self.dl_bn2 = nn.BatchNorm1d(64)
        self.dl_fc2 = nn.Linear(64,64)
        self.dl_bn3 = nn.BatchNorm1d(64)
        self.dl_fc3 = nn.Linear(64,256)


    def forward(self, x):
        # ====== SHORT FILTERS ======
        x = F.elu(self.sf_conv1(self.sf_conv1_bn(x)))
        x = F.elu(self.sf_conv2(self.sf_conv2_bn(x)))
        x = F.elu(self.sf_conv3(self.sf_conv3_bn(x)))

        # ====== MULTI FILTER MODS ======
        x = self.mf1_ap(x)
        x = self.mf1_bn(x)
        c1 = self.mf1_conv1(x)
        c2 = self.mf1_conv2(x)
        c3 = self.mf1_conv3(x)
        c4 = self.mf1_conv4(x)
        c5 = self.mf1_conv5(x)
        c6 = self.mf1_conv6(x)
        x = torch.cat((c1,c2,c3,c4,c5,c6), dim=3)
        x = self.mf1_conv_final(x)

        # SPEED UP BEGINNING TRAINGIN
        # x = self.mf2_ap(x)
        # x = self.mf2_bn(x)
        # c1 = self.mf2_conv1(x)
        # c2 = self.mf2_conv2(x)
        # c3 = self.mf2_conv3(x)
        # c4 = self.mf2_conv4(x)
        # c5 = self.mf2_conv5(x)
        # c6 = self.mf2_conv6(x)
        # x = torch.cat((c1,c2,c3,c4,c5,c6), dim=3)
        # x = self.mf2_conv_final(x)
        #
        # x = self.mf3_ap(x)
        # x = self.mf3_bn(x)
        # c1 = self.mf3_conv1(x)
        # c2 = self.mf3_conv2(x)
        # c3 = self.mf3_conv3(x)
        # c4 = self.mf3_conv4(x)
        # c5 = self.mf3_conv5(x)
        # c6 = self.mf3_conv6(x)
        # x = torch.cat((c1,c2,c3,c4,c5,c6), dim=3)
        # x = self.mf3_conv_final(x)

        # GPU not enough RAM
        # x = self.mf4_ap(x)
        # x = self.mf4_bn(x)
        # c1 = self.mf4_conv1(x)
        # c2 = self.mf4_conv2(x)
        # c3 = self.mf4_conv3(x)
        # c4 = self.mf4_conv4(x)
        # c5 = self.mf4_conv5(x)
        # c6 = self.mf4_conv6(x)
        # x = torch.cat((c1,c2,c3,c4,c5,c6), dim=3)
        # x = self.mf4_conv_final(x)

        # ====== DENSE LAYERS ======
        x = torch.flatten(x,1)
        x = self.dl_do(x)
        x = self.dl_bn1(x)
        x = F.elu(self.dl_fc1(x))
        x = self.dl_bn2(x)
        x = F.elu(self.dl_fc2(x))
        x = self.dl_bn3(x)
        x = self.dl_fc3(x)

        output = F.softmax(x,dim=1)
        # print('output', output.shape)
        return output


class MFMod(object):
    def __init__(self,m, input_chans):
        self.ap = nn.AvgPool2d((m,1))
        self.bn = nn.BatchNorm2d(input_chans)
        self.conv1 = nn.Conv2d(input_chans,24,(1,32))
        self.conv2 = nn.Conv2d(input_chans,24,(1,64))
        self.conv3 = nn.Conv2d(input_chans,24,(1,96))
        self.conv4 = nn.Conv2d(input_chans,24,(1,128))
        self.conv5 = nn.Conv2d(input_chans,24,(1,192))
        self.conv6 = nn.Conv2d(input_chans,24,(1,244)) # This is a single fullscale, cutoff by previous CNNs
        self.conv_final = nn.Conv2d(24,36,1)

class BadNet(nn.Module):
    def __init__(self):
        super(BadNet, self).__init__()

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
        # print('outs', outs, outs.shape)
        # print('targs', targets)

        # loss = F.nll_loss(outs, targets)
        loss = F.cross_entropy(outs, targets)
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
    pa.add_argument('-n', '--network', default='paper')
    return pa.parse_args()

def run():
    args = parse_args()

    print('Initializing Data Loaders...')
    tr_loader = DataLoader(FMASpectrogramsDataset(train=True),
                           batch_size=bsize, shuffle=True, num_workers=10, pin_memory=True)
    te_loader = DataLoader(FMASpectrogramsDataset(train=False),
                           batch_size=bsize, shuffle=True, num_workers=10, pin_memory=True)

    print('Initializing network and optimizer')
    if args.network == 'paper':
        net = PaperNet()
    else:
        net = BadNet()

    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    if cuda:
        net = net.cuda().half()

    # opt = optim.SGD(net.parameters(), lr=0.01)
    # default eps causes NaNs for 16bit training
    opt = optim.Adam(net.parameters(), lr=0.01, eps=1e-4)

    for e in range(1,epochs+1):
        print('Begining training at epoch {:d}'.format(e))
        train(net, tr_loader, device, opt)
        test(net, tr_loader, device)
        torch.save(net.state_dict(), 'saves/nn-ep-{:03d}.pt'.format(e))

    # Load like so:
    # newnet = Net()
    # newnet.load_state_dict(torch.load('nn.pt'))

if __name__ == '__main__':
    run()
