#!/usr/bin/env python3

import os
import argparse

import numpy as np

from KIPAC.nuXgal.Likelihood import Likelihood

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('f_astro', help='Factor multiplied by fraction of diffuse flux to inject as signal', type=float)
    parser.add_argument('N_yr', help='Number of years of synthetic data, choose 3 or 10', type=int)
    parser.add_argument('-o', '--output', help='Output file', default='eventmap.fits.gz', type=str)
    parser.add_argument('--use-csky', action='store_true')
    parser.add_argument('--compute-std', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    
    llh = Likelihood(args.N_yr, 'WISE', computeSTD=args.compute_std, Ebinmin=1, Ebinmax=4, lmin=50, use_csky=args.use_csky)
    llh.generate_data(args.f_astro, args.output, overwrite=args.overwrite)

if __name__ == '__main__':
    main()
