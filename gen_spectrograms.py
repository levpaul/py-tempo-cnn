#!/usr/bin/env python

import argparse
import glob, os
import librosa
import numpy as np

from multiprocessing import Pool as ThreadPool
from multiprocessing import cpu_count

win_len = 1024
output_dir = ''

def extract_spectro(f):
    global output_dir
    print('extracting spec of ', f)
    if not f.endswith('.mp3'):
        print('bad file')
        return

    y, sr = librosa.load(f, sr=11025)
    hop_len = win_len // 2
    data = librosa.feature.melspectrogram(y,sr,n_fft=win_len,hop_length=hop_len,power=1,n_mels=40,fmin=20,fmax=5000)
    data = data.astype(np.float16)
    print('saving spec of ', f)
    np.save('{}/{}.npy'.format(output_dir, os.path.basename(f)[:-4]), data)

def parse_args():
    pa = argparse.ArgumentParser()
    pa.add_argument('-f', '--folder', required=True, help='folder containing mp3s to convert to spectrograms')
    pa.add_argument('-o', '--output-folder', required=True, help='folder which spectrograms in .npy format will be saved')
    return pa.parse_args()

def run():
    global output_dir
    args = parse_args()
    output_dir = args.output_folder

    pool = ThreadPool(cpu_count())
    pool.map(extract_spectro, glob.iglob(args.folder + '/**', recursive=False))

if __name__ == '__main__':
    run()
