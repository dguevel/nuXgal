#!/usr/bin/env python3

import os
import argparse

import numpy as np

from KIPAC.nuXgal.Likelihood import Likelihood

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('f_astro', help='Factor multiplied by fraction of diffuse flux to inject as signal', type=float)
    parser.add_argument('N_yr', help='Number of years of synthetic data, choose 3 or 10', type=int)
    parser.add_argument('N_re', help='Number of trials', type=int)
    parser.add_argument('-o', '--output', help='Output file', default='fits.csv', type=str)
    parser.add_argument('--emin', default=1, type=int)
    parser.add_argument('--emax', default=4, type=int)
    parser.add_argument('--use-csky', action='store_true')
    parser.add_argument('--compute-std', action='store_true')
    parser.add_argument('--save-data', action='store_true')
    args = parser.parse_args()
    
    llh = Likelihood(args.N_yr, 'WISE', computeSTD=args.compute_std, Ebinmin=args.emin, Ebinmax=args.emax, lmin=50, use_csky=args.use_csky)
    fit = llh.get_many_fits(args.N_re, args.f_astro, save=args.save_data)

    data = np.concatenate([
        np.repeat(fit['f_astro_factor'], fit['TS'].size)[:,np.newaxis],
        fit['flux_fit'],
        np.repeat(fit['f_astro_inj'][np.newaxis,:], fit['TS'].size, axis=0),
        fit['f_astro_fit'],
        np.repeat(fit['N_astro_inj'][np.newaxis,:], fit['TS'].size, axis=0),
        fit['N_astro_fit'],
        fit['TS'][:,np.newaxis]
        ], axis=1)

    header = ['f_astro_factor']
    header += ['flux_fit_{}'.format(i) for i in range(args.emin, args.emax)]
    header += ['f_astro_inj_{}'.format(i) for i in range(args.emin, args.emax)]
    header += ['f_astro_fit_{}'.format(i) for i in range(args.emin, args.emax)]
    header += ['N_astro_inj_{}'.format(i) for i in range(args.emin, args.emax)]
    header += ['N_astro_fit_{}'.format(i) for i in range(args.emin, args.emax)]
    header += ['TS']


    with open(args.output, 'wb') as f:
        np.savetxt(f, data, delimiter=',', header=','.join(header), comments='')

if __name__ == '__main__':
    main()
