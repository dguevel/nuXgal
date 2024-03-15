"""Use nuXgal and csky to generate synthetic data sets"""

from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm
import json

from KIPAC.nuXgal.Likelihood import Likelihood
from KIPAC.nuXgal.NeutrinoSample import NeutrinoSample
from KIPAC.nuXgal import Defaults

def find_n_inj_per_bin(trial, ebinmin, ebinmax):
    """
    Find the number of injections per energy bin.

    Parameters:
        trial (list): The trial data.
        ebinmin (int): Minimum energy bin index.
        ebinmax (int): Maximum energy bin index.

    Returns:
        list: The number of injections per energy bin.
    """
    n_inj = []
    for i in range(ebinmin, ebinmax):
        elo = Defaults.map_logE_edge[i]
        ehi = Defaults.map_logE_edge[i + 1]
        n_inj.append(0)

        for yr in trial:
            if len(yr) > 1:
                idx = (yr[1]['log10energy'] > elo) * (yr[1]['log10energy'] < ehi)
                n_inj[-1] += int(idx.sum())

    return n_inj

def main():
    parser = ArgumentParser(description=main.__doc__)
    parser.add_argument('-i', '--inject', type=int, nargs='+',
                        help='Number of events to inject')
    parser.add_argument('-o', '--output',
                        help='Output file name')
    parser.add_argument('-n', '--n-trials', type=int, default=1,
                        help='Number of trials')
    parser.add_argument('-d', '--N-yr', type=str, default='nt_v5',
                        help='Data set name')
    parser.add_argument('-g', '--galaxy-catalog', type=str,
                        default='unWISE_z=0.4', help='Galaxy sample name')
    parser.add_argument('--ebinmin', type=int, default=0,
                        help='Minimum energy bin')
    parser.add_argument('--ebinmax', type=int, default=Defaults.NEbin,
                        help='Maximum energy bin')
    parser.add_argument('--gamma', type=float, default=2.5,
                        help='Spectral index')
    parser.add_argument('--fit-bounds', action='store_true',
                        help='Use fit bounds')
    parser.add_argument('--bootstrap-niter', type=int, default=100,
                        help='Number of bootstrap iterations')
    parser.add_argument('--save-countsmap', action='store_true',
                        help='Save counts map')
    parser.add_argument('--mcbg', action='store_true',
                        help='Use MC background')
    parser.add_argument('--save-cov', action='store_true',
                        help='Save covariance matrix')
    parser.add_argument('--recompute-model', action='store_true',
                        help='Recompute model')

    args = parser.parse_args()

    if args.fit_bounds is None:
        fit_bounds = None
    else:
        fit_bounds = 4 * [[0, 1], ]

    # Set up the likelihood
    llh = Likelihood(
        N_yr=args.N_yr,
        galaxyName=args.galaxy_catalog,
        Ebinmin=args.ebinmin,
        Ebinmax=args.ebinmax,
        lmin=1,
        gamma=args.gamma,
        fit_bounds=fit_bounds,
        mc_background=args.mcbg,
        recompute_model=args.recompute_model)

    result_list = []

    for ninj in args.inject:
        for i in tqdm(np.arange(args.n_trials)):
            trial, nexc = llh.event_generator.SyntheticTrial(
                ninj, keep_total_constant=False)
            ns = NeutrinoSample()
            ns.inputTrial(trial)
            llh.inputData(ns, bootstrap_niter=args.bootstrap_niter)

            result_dict = {}

            # save command line args to file
            for key in vars(args):
                result_dict[key] = getattr(args, key)
            result_dict['n_inj'] = ninj
            result_dict['n_total'] = int(np.sum(llh.Ncount))
            result_dict['n_total_i'] = dict(zip(range(args.ebinmin, args.ebinmax), llh.Ncount[args.ebinmin:args.ebinmax].astype(int).tolist()))
            result_dict['n_inj_i'] = dict(zip(range(args.ebinmin, args.ebinmax), find_n_inj_per_bin(trial, args.ebinmin, args.ebinmax)))
            result_dict['f_inj_i'] = dict(zip(range(args.ebinmin, args.ebinmax), (find_n_inj_per_bin(trial, args.ebinmin, args.ebinmax)/llh.Ncount[args.ebinmin:args.ebinmax]).tolist()))
            result_dict['f_inj'] = float(result_dict['n_inj'] / result_dict['n_total'])
            result_dict['flux_inj'] = llh.event_generator.trial_runner.to_dNdE(ninj, E0=1e5, gamma=2.5) / (4*np.pi*llh.f_sky)
            result_dict['n_to_flux'] = llh.event_generator.trial_runner.to_dNdE(1, E0=1e5, gamma=2.5) / (4*np.pi*llh.f_sky)
            result_dict['gamma_inj'] = args.gamma


            # save trial info to file
            if args.save_countsmap:
                result_dict['countsmap'] = ns.countsmap.tolist()
            result_dict['cls'] = {}
            for ebin in range(args.ebinmin, args.ebinmax):
                result_dict['cls'][ebin] = llh.w_data[ebin].tolist()

            # save std and cov matrix if bootstrapped
            if args.bootstrap_niter > 0:
                result_dict['w_std'] = {}
                if args.save_cov:
                    result_dict['w_cov'] = {}
                for ebin in range(args.ebinmin, args.ebinmax):
                    if args.save_cov:
                        result_dict['w_cov'][ebin] = llh.w_cov[ebin].tolist()
                    result_dict['w_std'][ebin] = llh.w_std[ebin].tolist()

            result_list.append(result_dict)

    with open(args.output, 'w') as f:
        json.dump(result_list, f, indent=4)


if __name__ == '__main__':
    main()
