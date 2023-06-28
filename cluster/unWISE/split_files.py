#!/usr/bin/env python3

from glob import glob
from argparse import ArgumentParser
import os

import numpy as np

def main():
    parser = ArgumentParser('Split a list of files into sublists for cluster use. \n eg: python git/nuXgal/cluster/unWISE/split_files.py -i $(find /data/user/dguevel/unWISE/unwise/release/band-merged/ -name "*.fits") -n 100 -o /home/dguevel/git/nuXgal/cluster/unWISE/file_lists')
    parser.add_argument('-i', '--input', type=str, nargs='+', help='Input file list')
    parser.add_argument('-n', '--n-files', default=100, type=int, help='Number of lists to split files into')
    parser.add_argument('-o', '--output-path', help='Output file path')
    args = parser.parse_args()

    files_split = []
    files = list(args.input)
    while len(files) > 0:
        files_split.append(files[:args.n_files])
        del files[:args.n_files]

    for i, fs in enumerate(files_split):
        outfile = os.path.join(args.output_path, 'list_{}.txt'.format(str(i)))
        with open(outfile, 'w') as of:
            of.write('\n'.join(fs))

if __name__ == '__main__':
    main()