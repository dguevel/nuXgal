#!/usr/bin/env python3

import os
import argparse

import numpy as np

from KIPAC.nuXgal.Likelihood import Likelihood

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('N_yr', help='Number of years of synthetic data, choose 3 or 10', type=int)
    parser.add_argument('N_re', help='Number of trials', type=int)
    parser.add_argument('-o', '--output', help='Output file name', default='upper_limits.csv', type=str)
    parser.add_argument('--emin', default=1, type=int)
    parser.add_argument('--emax', default=4, type=int)
    parser.add_argument('--use-csky', action='store_true')
    args = parser.parse_args()
    
    llh = Likelihood(args.N_yr, 'WISE', computeSTD=False, Ebinmin=args.emin, Ebinmax=args.emax, lmin=50, use_csky=args.use_csky)
    ul = llh.upperLimit(args.N_re)

    data = np.concatenate([ul['flux'], ul['f_astro'], ul['n_astro'], ul['TS'][:, np.newaxis]], axis=1)
    header = ['flux_{}'.format(i) for i in range(args.emin, args.emax)]
    header += ['f_astro_{}'.format(i) for i in range(args.emin, args.emax)]
    header += ['n_astro_{}'.format(i) for i in range(args.emin, args.emax)]
    header += ['TS']

    with open(args.output, 'wb') as f:
        np.savetxt(f, data, delimiter=',', header=','.join(header), comments='')

if __name__ == '__main__':
    main()
