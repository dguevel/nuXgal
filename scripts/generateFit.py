#!/usr/bin/env python3

import os
import argparse

import numpy as np

from KIPAC.nuXgal.Likelihood import Likelihood

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('n_inj', help='Factor multiplied by fraction of diffuse flux to inject as signal', type=float)
    parser.add_argument('N_yr', help='Number of years of synthetic data, choose 3 or 10', type=int)
    parser.add_argument('N_re', help='Number of trials', type=int)
    parser.add_argument('-o', '--output', help='Directory for output files', default=os.path.join('data', 'user', 'dguevel', 'TS'), type=str)
    parser.add_argument('--emin', default=1, type=int)
    parser.add_argument('--emax', default=4, type=int)
    parser.add_argument('--use-csky', action='store_true')
    args = parser.parse_args()
    
    llh = Likelihood(args.N_yr, 'WISE', computeSTD=False, Ebinmin=args.emin, Ebinmax=args.emax, lmin=50, use_csky=args.use_csky)
    result = llh.sensitivitySamples(args.N_re, args.f_astro, writeData=False, return_n_inj=True)

    TS_path = os.path.join(args.output, 'TS_{f_astro}_{n_yr}yr.txt'.format(f_astro=args.f_astro, n_yr=args.N_yr))

    with open(TS_path, 'wb') as f:
        np.savetxt(f, TS)

if __name__ == '__main__':
    main()
