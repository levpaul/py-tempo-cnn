#!/usr/bin/env python

import argparse
from os import walk, path
import librosa
import numpy as np

win_len = 1024

def extract_tempogram(file):
    y, sr = librosa.load(file, sr=11025)
    hop_len = win_len // 2
    data = librosa.feature.melspectrogram(y,sr,n_fft=win_len,hop_length=hop_len,power=1,n_mels=40,fmin=20,fmax=5000)
    return data.astype(np.float16)

def extract_from_dir(dirname, output):
    print('Looking for mp3s...')

    for (dpath, _, fnames) in walk(dirname):
        for file in fnames:
            try:
                if not file.endswith('.mp3'):
                    continue
                tg = extract_tempogram(path.join(dpath,file))
                np.save('{}/{}.npy'.format(output, file[:-4]), tg)
            except:
                continue

    print('Done walking')

def parse_args():
    pa = argparse.ArgumentParser()
    pa.add_argument('-f', '--folder', required=True)
    pa.add_argument('-o', '--output-folder', required=True)
    return pa.parse_args()

def run():
    args = parse_args()
    extract_from_dir(args.folder, args.output_folder)

if __name__ == '__main__':
    run()