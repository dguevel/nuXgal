#!/usr/bin/env python3

import os
import argparse

import numpy as np

from KIPAC.nuXgal.Likelihood import Likelihood

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('N_yr', help='Number of years of synthetic data, choose 3 or 10', type=int)
    parser.add_argument('N_re', help='Number of trials', type=int)
    parser.add_argument('-o', '--output', help='Director for output files', default=os.path.join('data', 'user', 'dguevel', 'UL'), type=str)
    parser.add_argument('--emin', default=1, type=int)
    parser.add_argument('--emax', default=4, type=int)
    parser.add_argument('--use-csky', action='store_true')
    args = parser.parse_args()
    
    llh = Likelihood(args.N_yr, 'WISE', computeSTD=False, Ebinmin=args.emin, Ebinmax=args.emax, lmin=50, use_csky=args.use_csky)
    flux, f_astro, N_astro = llh.upperLimit(args.N_re)

    flux_path = os.path.join(args.output, 'UL_flux.txt')
    f_astro_path = os.path.join(args.output, 'UL_f_astro.txt')
    N_astro_path = os.path.join(args.output, 'UL_N_astro.txt')

    with open(flux_path, 'wb') as f:
        np.savetxt(f, flux)

    with open(f_astro_path, 'wb') as f:
        np.savetxt(f, f_astro)

    with open(N_astro_path, 'wb') as f:
        np.savetxt(f, N_astro)

if __name__ == '__main__':
    main()
