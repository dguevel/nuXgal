"""Calculate the covariance matrices from trials"""
import numpy as np
import json
import os
from argparse import ArgumentParser
import matplotlib.pyplot as plt

from tqdm import tqdm

def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", nargs="+", help="JSON files to load")
    parser.add_argument("-o", "--output", help="Output file name", default='w_cov.npy')
    parser.add_argument("--plot", action="store_true", help="Plot the covariance matrix")

    args = parser.parse_args()

    # read in the data
    offset = 0
    first = True
    for f in tqdm(args.input):
        with open(f) as f:
            data = json.load(f)

            # assume the first file is the same as the rest
            if first:
                n_ebin = len(data[0]['cls'])
                n_cl = len(data[0]['cls']['0'])
                cl_matrix = np.zeros((n_ebin, len(data)*len(args.input), n_cl))
                first = False

            for d in data:
                for i in range(3):
                    cl_matrix[i][offset] = np.array(d['cls'][str(i)])
                offset += 1

    # calculate the covariance matrix
    cov_matrix = np.zeros((n_ebin, n_cl, n_cl))
    corr_matrix = np.zeros((n_ebin, n_cl, n_cl))
    for i in range(n_ebin):
        cov_matrix[i] = np.cov(cl_matrix[i].T)
        corr_matrix[i] = np.corrcoef(cl_matrix[i].T)

    # save the covariance matrix
    np.save(args.output, cov_matrix)

    if args.plot:
        output = args.output.replace('.npy', '.png')
        plot_cov(cov_matrix, corr_matrix, output)


def plot_cov(cov_matrix, corr_matrix, output):

    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    ell = np.arange(cov_matrix.shape[1])
    for i in range(cov_matrix.shape[0]):
        im = ax[0, i].pcolor(ell, ell, cov_matrix[i], vmax=1e-7, vmin=0)
        plt.colorbar(im, ax=ax[0, i])
        ax[0, i].set_title(r"Corr$(\ell_1,~\ell_2)$: energy bin {0:d}".format(i))

    for i in range(cov_matrix.shape[0]):
        im = ax[1, i].pcolor(ell, ell, corr_matrix[i], vmax=1, vmin=0)
        plt.colorbar(im, ax=ax[1, i])
        ax[1, i].set_title(r"Cov$(\ell_1,~\ell_2)$: energy bin {0:d}".format(i))
        ax[1, i].set_xlabel(r"$\ell_1$")
    ax[0, 0].set_ylabel(r"$\ell_2$")
    ax[1, 0].set_ylabel(r"$\ell_2$")

    plt.savefig(output)
    plt.close()

    print(f"Saved plot to {output}")


if __name__ == '__main__':
    main()