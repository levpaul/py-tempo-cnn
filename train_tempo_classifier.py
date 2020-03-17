#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse
import re
import glob

from os import path, remove
from pathlib import Path

from lib.datasets.spectrograms import SpectrogramsDataset

epochs=10000
bsize=256
spectro_window_size = 256

DEFAULT_SAVE_FORMAT = 'default'
DEFAULT_LOAD_FORMAT = 'default'


def parse_args():
    pa = argparse.ArgumentParser()

    pa.add_argument('-n', '--network', default='tempo', type=str,
                    help='Select NN to use, can chose from [tempo,basic]')
    pa.add_argument('-d', '--data-dir', required=True, type=str,
                    help='Data containing .npy files for training and test data and labels')
    pa.add_argument('-r', '--learning-rate', default=0.01, type=float,
                    help='Set learning rate of optimizer')
    pa.add_argument('-l', '--load', nargs='?', const=DEFAULT_LOAD_FORMAT, type=str,
                    help='Resume training from a saved model - specify custom filename or use the default')
    pa.add_argument('-s', '--save-name', default=DEFAULT_SAVE_FORMAT, type=str,
                    help='Specify a custom filename format for saved models (will have ep-####.pt appended)')

    return pa.parse_args()


def train(net, loader, device, opt, criterion):
    net.train()
    correct = 0
    for bidx, (data, targets) in enumerate(loader):
        data, targets = data.to(device), targets.to(device)
        opt.zero_grad()

        out = net(data)

        loss_vec = criterion(out, targets)
        loss_vec.backward() # <- compute backprop derivs

        # Compute acc
        pred = out.argmax(dim=1, keepdim=True)
        correct += pred.eq(targets.view_as(pred)).sum().item()

        opt.step()

    return 100. * correct / len(loader.dataset)


def test(net, loader, device):
    net.eval()
    correct = 0
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            out = net(data)
            pred = out.argmax(dim=1, keepdim=True)
            correct += pred.eq(targets.view_as(pred)).sum().item()

    return 100. * correct / len(loader.dataset)


def run():
    args = parse_args()

    print('Initializing Data Loaders...')
    tr_loader = DataLoader(SpectrogramsDataset(train=True, dir=args.data_dir),
                           batch_size=bsize, shuffle=True, num_workers=1, pin_memory=True)
    te_loader = DataLoader(SpectrogramsDataset(train=False, dir=args.data_dir),
                           batch_size=bsize, shuffle=True, num_workers=1, pin_memory=True)

    print('Initializing network and optimizer')
    if args.network == 'tempo':
        from lib.nets.tempo_net import TempoNet
        net = TempoNet()
    elif args.network == 'basic':
        from lib.nets.basic_net import BasicNet
        BasicNet()
    else:
        print('Invalid neural network; "{}" is not supported'.format(args.network))
        exit(1)

    # Save template
    save_dir = path.join(args.data_dir, 'saves')
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    if args.save_name == DEFAULT_SAVE_FORMAT:
        save_name_base= save_dir+'/nn-{:s}-lr{:f}'.format(args.network, args.learning_rate)
    else:
        save_name_base = save_dir+'/'+args.save_name
    epoch_save_format = save_name_base+'-ep{:04d}.pt'

    # Load template
    start_epoch = 1
    if args.load:
        if args.load == DEFAULT_LOAD_FORMAT:
            load_name_base = save_dir + '/nn-{:s}-lr{:f}'.format(args.network, args.learning_rate)
        else:
            load_name_base = save_dir + '/' + args.load
        loads = glob.iglob(load_name_base + '-ep*.pt')

        max_ep = 0
        max_f = None
        for f in loads:
            f_base = path.basename(f)
            re_results = re.search('.*-ep(\d*).pt$', f_base)
            if re_results is None or len(re_results.groups()) != 1:
                continue
            f_ep = int(re_results.groups()[0])
            if f_ep > max_ep:
                max_ep = f_ep
                max_f = f

        if max_f is None:
            print('Invalid load file - please ensure a previous save exists for the given load format')
            exit(1)
        else:
            print('Found previous model to load at epoch {:d}, model: {:s}'.format(max_ep, path.basename(max_f)))

        start_epoch = max_ep+1
        net.load_state_dict(torch.load(max_f))

    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0") if cuda else "cpu"
    if cuda:
        net = net.cuda().half()

    opt = optim.SGD(net.parameters(), lr=args.learning_rate)
    # opt = optim.Adam(net.parameters(), lr=0.01, eps=1e-4) # <- deconverged to NaNs at 2.5 epochs
    criterion = nn.NLLLoss()
    # criterion = nn.CrossEntropyLoss()

    print('ep,train_acc,test_acc')
    for e in range(start_epoch,epochs+1):
        tr_acc = train(net, tr_loader, device, opt, criterion)
        te_acc = test(net, te_loader, device)
        print('{:d},{:.2f},{:.2f}'.format(e, tr_acc, te_acc))
        torch.save(net.state_dict(), epoch_save_format.format(e))
        if e > 1:
            remove(epoch_save_format.format(e-1))

if __name__ == '__main__':
    run()
