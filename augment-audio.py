#!/usr/bin/env python


import string
import random
import argparse

from multiprocessing.dummy import Pool as ThreadPoolpool
from multiprocessing import cpu_count

# ========================================================
#   Arg Parsing
# ========================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Data augmentation script - pass in a directory which contains a test or train (or both) directories containing audio (wav) files.')
    parser.add_argument('-n', '--end-dataset-size', type=int, default=100000, help='Specify the total number of files to generate')
    parser.add_argument('-s', '--source-dir', required=True, help='The directory the train and or test directories are in')
    parser.add_argument('-o', '--output-dir', required=True, help='The directory to use for outputting augmented data into')
    parser.add_argument('-t', '--augment-type', choices=['train', 'test', 'both'], default='both', help='The type of datasets to compile')
    parser.add_argument('-j', '--number-workers', type=int, default=cpu_count(), help='The number of workers to use for parallel processing')
    return parser.parse_args()

# ========================================================
# No diff between poolings except normal Pool gives Exceptions at runtime

def augment_data(source_dir, out_dir, train_set, test_set, num_workers):
    print('To implement')

def gen_spectrograms(source_dir, out_dir, train_set, test_set, num_workers):
    print('To implement')

def squash_to_dataset(source_dir, out_dir, train_set, test_set, num_workers):
    print('To implement')


def run():
    args = parse_args()
    train_set = True if args.augment_type == 'both' or args.augment_type == 'train' else False
    test_set = True if args.augment_type == 'both' or args.augment_type == 'test' else False

    augment_data(source_dir=args.source_dir, out_dir=args.output_dir, train_set=train_set, test_set=test_set, num_workers=args.number_workers)
    gen_spectrograms(source_dir=args.source_dir, out_dir=args.output_dir, train_set=train_set, test_set=test_set, num_workers=args.number_workers)
    squash_to_dataset(source_dir=args.source_dir, out_dir=args.output_dir, train_set=train_set, test_set=test_set, num_workers=args.number_workers)
    # augmenter = DataAugmenter(file_names, args.end_dataset_size)
    # pool.map(transform, augmenter)

if __name__ == '__main__':
    run()
