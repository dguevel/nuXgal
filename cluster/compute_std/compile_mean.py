from argparse import ArgumentParser
import json
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from KIPAC.nuXgal import Defaults

def main():
    parser = ArgumentParser(description='Compute standard deviation of cross power spectrum')
    parser.add_argument('-i', '--input', help='Input files')
    parser.add_argument('-o', '--output', help='Output file')
    parser.add_argument('--n-ebins', default=3, type=int, help='Number of energy bins')
    args = parser.parse_args()

    # Read in data
    data = []
    files = glob(args.input)
    for filename in files:
        with open(filename, 'r') as f:
            data.extend(json.load(f))

    # Compute mean
    cl_matrix = np.zeros((len(data), args.n_ebins, Defaults.NCL))
    for i, d in enumerate(data):
        for ebin in d['cls']:
            cl_matrix[i, int(ebin)] = d['cls'][ebin]

    cl_mean = np.mean(cl_matrix, axis=0)

    fig, ax = plt.subplots(dpi=150)
    l = np.arange(Defaults.NCL)
    lmin = 0
    for i in range(3):
        ax.plot(l[lmin:], l[lmin:]**2*cl_mean[i, lmin:])
    plt.savefig('plots/mean.png')

    pass





if __name__ == '__main__':
    main()