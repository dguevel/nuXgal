#!/usr/bin/env python3

import os
import argparse
import json

import numpy as np
import healpy as hp

from KIPAC.nuXgal.Likelihood import Likelihood
from KIPAC.nuXgal.NeutrinoSample import NeutrinoSample

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Input file', type=str, nargs='+')
    parser.add_argument('-o', '--output', help='Output file', type=str, nargs='+')
    parser.add_argument('--emin', default=1, type=int)
    parser.add_argument('--emax', default=4, type=int)
    parser.add_argument('--lmin', default=50, type=int)
    parser.add_argument('--no-csky', action='store_true')
    args = parser.parse_args()

    results = []

    for fname, outname in zip(args.input, args.output):
        datamap, header = hp.read_map(fname, None, h=True)
        header = dict(header)
        
        llh = Likelihood(header['N_YR'], 'WISE', computeSTD=False, Ebinmin=args.emin, Ebinmax=args.emax, lmin=args.lmin, use_csky=not args.no_csky)
        ns = NeutrinoSample()
        ns.inputCountsmap(datamap)
        llh.inputData(ns)

        f_astro_fit, TS = llh.minimize__lnL()
        TS_ebin = np.zeros(f_astro_fit.size)
        for i, ebin in enumerate(range(args.emin, args.emax)):
            TS_ebin[i] = 2 * (llh.log_likelihood_Ebin(f_astro_fit[i], ebin) - llh.log_likelihood_Ebin(0, ebin))

        n_astro_fit = llh.Ncount[args.emin: args.emax] * f_astro_fit
        f_astro_ul = llh.upperLimit()
        n_astro_ul = llh.Ncount[args.emin: args.emax] * f_astro_ul

        f_astro_inj = np.array([header['F_INJ_{}'.format(j)] for j in range(args.emin, args.emax)])
        n_astro_inj = llh.Ncount[args.emin: args.emax] * f_astro_inj
        
        results.append({
            'file': fname,
            'emin': args.emin,
            'emax': args.emax,
            'f_astro_fit': f_astro_fit.tolist(),
            'f_astro_upper_limit': f_astro_ul.tolist(),
            'f_astro_inj': f_astro_inj.tolist(),
            'n_astro_fit': n_astro_fit.tolist(),
            'n_astro_upper_limit': n_astro_ul.tolist(),
            'n_astro_inj': n_astro_inj.tolist(),
            'TS': TS,
            'TS_ebin': TS_ebin.tolist(),
            'N_yr': header['N_YR'],
            'csky': True,
        })
        
        with open(outname, 'w') as out:
            json.dump(results, out)

if __name__ == '__main__':
    main()
