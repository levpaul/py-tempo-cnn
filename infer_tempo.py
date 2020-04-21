#!/usr/bin/env python

import torch
from torch.utils.data import DataLoader

from lib.nets.tempo_net import TempoNet
from lib.datasets.spectrograms import RawFileDataset

import argparse

import matplotlib
import matplotlib.pyplot as plt

def parse_args():
    pa = argparse.ArgumentParser()

    pa.add_argument('-m', '--model', required=True, type=str,
                    help='Location of .pt model file')
    pa.add_argument('file', nargs=argparse.REMAINDER, type=str)

    args = pa.parse_args()

    if len(args.file) != 1:
        print('Received {} file args; expected only 1!'.format(len(args.file)))
        exit(1)

    args.file = args.file[0]

    return args


def eval(net, loader, device):
    net.eval()
    results = []
    with torch.no_grad():
        for spectro in loader:
            out = net(spectro.to(device))
            pred = out.argmax(dim=1, keepdim=True)[0][0]
            print('pred: {}'.format(pred))
            results.append(pred)

    plt.plot(range(len(loader)), results)
    plt.show()

def run():
    args = parse_args()

    print('Initializing Data Loaders...')
    loader = DataLoader(RawFileDataset(file=args.file),
                           batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    # ==============
    # Load template
    # ==============
    t_model = torch.load(args.model)
    mfmod_level = 1
    if 'mf2_bn.weight' in t_model.keys():
        mfmod_level = 2
    if 'mf3_bn.weight' in t_model.keys():
        mfmod_level = 3
    if 'mf4_bn.weight' in t_model.keys():
        mfmod_level = 4
    net = TempoNet(mfmod_levels=mfmod_level)
    net.load_state_dict(torch.load(args.model))

    # ==============
    # Prepare CUDA
    # ==============
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0") if cuda else "cpu"
    if cuda:
        net = net.cuda().half()

    # ==============
    # Eval
    # ==============
    eval(net, loader, device)

if __name__ == '__main__':
    run()
