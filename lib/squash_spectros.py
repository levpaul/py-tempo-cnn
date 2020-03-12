#!/usr/bin/env python

import argparse
import glob
import numpy as np

def squash_spectros(folder, output_filename):
    count = 0
    max = 0
    arrays = []
    label_tempos = []
    label_lengths = []

    for f in glob.iglob(folder + '/*.npy', recursive=False):
        if count % 10000 == 0:
            print('Current count',count)
        if not f.endswith('.npy'):
            print('bad file', f)
            return
        n = np.load(f)
        l = n.shape[1]
        arrays.append(n)
        if l > max:
            max = l

        # Labels
        label_tempos.append(int(f[:-7].split('-')[-1]))
        label_lengths.append(l)

        count+= 1

    ndata = np.empty((count,1,40,max), dtype=np.float16)
    nlabels = np.array((label_tempos, label_lengths))

    for c in range(count):
        ndata[c,0,:,:arrays[c].shape[1]] = arrays[c]
        arrays[c] = 0

    print('data dims: ', ndata.shape)
    print('label dims: ', ndata.shape)

    np.save(output_filename + '-data.npy',ndata)
    np.save(output_filename + '-labels.npy',nlabels)

def parse_args():
    pa = argparse.ArgumentParser(usage='Convert a directory of spectrograms to a pair of data/label .npy files. Spectrograms can be of variable length, so the label npy file contains bpm and then datasize per spectrogram')
    pa.add_argument('-f', '--folder', required=True, help='folder which the spectrogram npy files exist - all will be taken')
    pa.add_argument('-o', '--output-folder', required=True, help='the filename of the output npy files - will have {labels|data} appended for a total of 2 files')
    return pa.parse_args()

def run():
    global output_dir
    args = parse_args()
    output_dir = args.output_folder

    squash_spectros(args.folder, args.output_folder)

if __name__ == '__main__':
    run()
