#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse
import random
import re
from torch.utils.data import Dataset, DataLoader
from os import path

epochs=10000
bsize=256
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

        # output = F.softmax(x,dim=1)
        output = F.log_softmax(x,dim=1)
        # print('output', output.shape)
        return output

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
        # output = F.softmax(x, dim=1)
        output = F.log_softmax(x, dim=1)
        return output

class SpectrogramsDataset(Dataset):
    def __init__(self, dir, train):
        if train:
            self.data = np.load(path.join(dir,'train-data.npy'))
            self.labels = np.load(path.join(dir,'train-labels.npy'))
        else:
            self.data = np.load(path.join(dir, 'test-data.npy'))
            self.labels = np.load(path.join(dir, 'test-labels.npy'))

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return (self.data[idx], self.labels[idx])

def train(net, loader, device, opt, criterion):
    net.train()
    # running_loss = 0.
    correct = 0
    for bidx, (data, targets) in enumerate(loader):
        data, targets = data.to(device), targets.to(device)
        opt.zero_grad()

        out = net(data)

        # loss = F.nll_loss(outs, targets)
        # loss = F.cross_entropy(outs, targets)
        loss_vec = criterion(out, targets)
        # running_loss += loss_vec.item()
        loss_vec.backward() # <- compute backprop derivs

        # Compute acc
        pred = out.argmax(dim=1, keepdim=True)
        correct += pred.eq(targets.view_as(pred)).sum().item()

        opt.step()

        # if bidx % 5000 == 49:
        #     print('[{:5d}] loss: {:0.3f}'.format(bidx+1, running_loss/50))
        #     running_loss = 0.
    return 100. * correct / len(loader.dataset)

def test(net, loader, device, criterion):
    net.eval()
    # loss = 0
    correct = 0
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            out = net(data)
            pred = out.argmax(dim=1, keepdim=True)
            # loss += criterion(out, targets).item()
            correct += pred.eq(targets.view_as(pred)).sum().item()

    return 100. * correct / len(loader.dataset)
    # loss /= len(loader.dataset)
    # return loss
    # print("Test set: Average loss: {:.4f}, Acc: {}/{} ({:.0f}%)\n".format(
    #     loss, correct, len(loader.dataset),
    #     100. * correct / len(loader.dataset)
    # ))

def parse_args():
    pa = argparse.ArgumentParser()
    pa.add_argument('-n', '--network', default='ssa', help='Select NN to use, can chose from [ssa,bad]')
    pa.add_argument('-r', '--resume', default=False, help='Resume training from a saved model, expected file name format nn-$network-$epoch.pt')
    pa.add_argument('-d', '--data-dir', required=True, help='Data containing .npy files for training and test data and labels')
    return pa.parse_args()

def run():
    args = parse_args()

    print('Initializing Data Loaders...')
    tr_loader = DataLoader(SpectrogramsDataset(train=True, dir=args.data_dir),
                           batch_size=bsize, shuffle=True, num_workers=1, pin_memory=True)
    te_loader = DataLoader(SpectrogramsDataset(train=False, dir=args.data_dir),
                           batch_size=bsize, shuffle=True, num_workers=1, pin_memory=True)

    if args.resume:
        resume_base = path.basename(args.resume)
        re_results = re.search('nn-(.*)-(\d*).pt$', resume_base)
        if re_results == None or len(re_results.groups()) != 2:
            print('Invalid resume file - please pass in as nn-$network-$epoch.pt')
            exit(1)
        args.network = re_results.groups()[0]
        start_epoch = int(re_results.groups()[1]) + 1
    else:
        start_epoch = 1

    print('Initializing network and optimizer')
    if args.network == 'ssa':
        net = PaperNet()
    elif args.network == 'bad':
        net = BadNet()
    else:
        print('Invalid neural network; "{}" is not supported'.format(args.network))
        exit(1)

    if args.resume:
        net.load_state_dict(torch.load(args.resume))

    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0") if cuda else "cpu"
    if cuda:
        net = net.cuda().half()

    opt = optim.SGD(net.parameters(), lr=0.001)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()
    # default eps causes NaNs for 16bit training
    # opt = optim.Adam(net.parameters(), lr=0.01, eps=1e-4) # <- deconverged to NaNs at 2.5 epochs

    print('ep,train_acc,test_acc')
    for e in range(start_epoch,epochs+1):
        tr_acc = train(net, tr_loader, device, opt, criterion)
        te_acc = test(net, te_loader, device, criterion)
        print('{:d},{:.2f},{:.2f}'.format(e, tr_acc, te_acc))
        torch.save(net.state_dict(), 'saves/nn-{:s}-{:04d}.pt'.format(args.network, e))

    # Load like so:
    # newnet = Net()
    # newnet.load_state_dict(torch.load('nn.pt'))

if __name__ == '__main__':
    run()
