#!/usr/bin/env python

import argparse
import os
from pathlib import Path
import math
import random
import string
import glob
import logging

from multiprocessing import cpu_count
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool

from lib.data_augmenter import DataAugmenter, transform
from lib.gen_spectrograms import extract_spectro
from lib.squash_spectros import squash_spectros

# ========================================================
#   Arg Parsing
# ========================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Data augmentation script - pass in a directory which contains a test and train directory containing audio (wav) files.')
    parser.add_argument('-n', '--end-dataset-size', type=int, default=100000, help='Specify the total number of files to generate')
    parser.add_argument('-s', '--source-dir', required=True, help='The directory the train and test directories are in')
    parser.add_argument('-o', '--output-dir', required=True, help='The directory to use for outputting augmented data into')
    parser.add_argument('-j', '--number-workers', type=int, default=cpu_count(), help='The number of workers to use for parallel processing')
    return parser.parse_args()

# ========================================================
# No diff between poolings except normal Pool gives Exceptions at runtime

def augment_data(n, source_dir, out_dir, num_workers):
    Path(os.path.join(out_dir, 'train/wavs')).mkdir(parents=True, exist_ok=False)
    Path(os.path.join(out_dir, 'test/wavs')).mkdir(parents=True, exist_ok=False)
    # Create data augmentor
    train_augmentor = DataAugmenter(source_dir=os.path.join(source_dir, 'train'), output_dir=os.path.join(out_dir, 'train/wavs'), n_times=math.ceil(n*0.9))
    test_augmentor = DataAugmenter(source_dir=os.path.join(source_dir, 'test'), output_dir=os.path.join(out_dir, 'test/wavs'), n_times=math.floor(n*0.1))
    print('Generating training wav files')
    pool = ThreadPool(num_workers)
    pool.starmap(transform, train_augmentor)
    print('DONE')
    print('Generating testing wav files')
    pool.starmap(transform, test_augmentor)
    print('DONE')
    pool.close()

def gen_spectrograms(work_dir, num_workers):
    Path(os.path.join(work_dir, 'train/specs')).mkdir(parents=True, exist_ok=False)
    Path(os.path.join(work_dir, 'test/specs')).mkdir(parents=True, exist_ok=False)

    pool = Pool(num_workers)

    print('Extracting training spectrograms')
    pool.map(extract_spectro, glob.iglob(work_dir+'/train/wavs/*.wav', recursive=False))
    print('DONE')

    print('Extracting training spectrograms')
    pool.map(extract_spectro, glob.iglob(work_dir+'/test/wavs/*.wav', recursive=False))
    print('DONE')

def squash_to_dataset(n, work_dir, out_dir):
    print('Squashing training spectrograms')
    squash_spectros(work_dir+'/train/specs', out_dir+'/train', math.ceil(n*0.9))
    print('DONE')
    print('Squashing testing spectrograms')
    squash_spectros(work_dir+'/test/specs', out_dir+'/test', math.floor(n*0.1))
    print('DONE')

def run():
    args = parse_args()

    aug_workdir = os.path.join(args.output_dir, 'augmented-' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=8)))

    augment_data(n=args.end_dataset_size, source_dir=args.source_dir, out_dir=aug_workdir, num_workers=args.number_workers)
    gen_spectrograms(work_dir=aug_workdir, num_workers=args.number_workers)
    squash_to_dataset(n=args.end_dataset_size, work_dir=aug_workdir, out_dir=args.output_dir)

if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    run()
