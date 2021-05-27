#!/usr/bin/env python3

import os
import argparse

import numpy as np

from KIPAC.nuXgal.Likelihood import Likelihood
from KIPAC.nuXgal.Defaults import SYNTHETIC_TS_SIGNAL_FORMAT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('N_yr', help='Number of years of synthetic data, choose 3 or 10', type=int)
    parser.add_argument('f_astro', help='Fraction of diffuse flux to inject', type=float)
    parser.add_argument('N_re', help='Number of trials', type=int)
    parser.add_argument('-o', '--output', help='Director for output files', default=os.path.join('data', 'user', 'dguevel', 'TS'), type=str)
    parser.add_argument('--emin', default=1)
    parser.add_argument('--emax', default=4)
    parser.add_argument('--use-csky', action='store_true')
    args = parser.parse_args()
    
    llh = Likelihood(args.N_yr, 'WISE', computeSTD=False, Ebinmin=args.emin, Ebinmax=args.emax, lmin=50, use_csky=args.use_csky)
    TS, n_inj = llh.TS_distribution(100, args.f_astro, writeData=False, return_n_inj=True)

    #TSpath = SYNTHETIC_TS_SIGNAL_FORMAT.format(f_diff=str(f_diff), galaxyName=self.gs.galaxyName, nyear=str(self.N_yr), astroModel=astroModel)
    if args.emin != args.emax:
        TSpath = os.path.join(args.output, 'TS_{n_inj}_{emin}-{emax}_WISE.txt'.format(n_inj=str(int(n_inj)), emin=str(args.emin), emax=str(args.emax)))
    else:
        TSpath = os.path.join(args.output, 'TS_{n_inj}_{emin}_WISE.txt'.format(n_inj=str(int(n_inj)), emin=str(args.emin), emax=str(args.emax)))

    with open(TSpath, 'ab') as f:
        np.savetxt(f, TS)


if __name__ == '__main__':
    main()
