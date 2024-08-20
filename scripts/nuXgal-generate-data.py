"""Use nuXgal and csky to generate synthetic data sets"""

from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm
import json
import csky as cy
import pandas as pd

from KIPAC.nuXgal.Likelihood import Likelihood
from KIPAC.nuXgal.NeutrinoSample import NeutrinoSample
from KIPAC.nuXgal import Defaults
from KIPAC.nuXgal.GalaxySample import GALAXY_LIBRARY
from KIPAC.nuXgal.CskyEventGenerator import CskyEventGenerator

def calc_f_inj(llh, ninj, trial):
    ninj_i = find_n_inj_per_bin(trial, llh.Ebinmin, llh.Ebinmax)
    if not (llh.acceptance is None):
        flux_inj = np.sum(((llh.event_generator.density_nu / llh.acceptance).T * ninj_i).T, axis=0)
        flux_total = np.sum(llh.neutrino_sample.countsmap / llh.acceptance, axis=0)
        f_inj = flux_inj.sum() / flux_total.sum()
    else:
        f_inj = np.sum(ninj_i) / np.sum(llh.Ncount)
    return f_inj

def point_source_trial_runner(ra, dec, gamma, ana):
    src = cy.sources(ra, dec, deg=True)
    cy.CONF['ana'] = ana
    cy.CONF['flux'] = cy.hyp.PowerLawFlux(gamma=gamma)
    trial_runner = cy.get_trial_runner(src=src)
    return trial_runner

def append_signal_trials(trial1, trial2):
    """Append signal events from trial 2 to trial 1."""
    n_seasons = len(trial1)
    for i in range(n_seasons):
        if len(trial2[i]) > 1:
            if len(trial1[i]) > 1:
                trial1[i][1]['energy'] = 10**trial1[i][1]['log10energy']
                trial2[i][1]['energy'] = 10**trial2[i][1]['log10energy']
                trial1[i][1] = cy.utils.Events.concatenate((trial1[i][1], trial2[i][1]))
            else:
                trial1[i].append(trial2[i][1])
    return trial1

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
    parser.add_argument('--bootstrap-niter', type=int, default=0,
                        help='Number of bootstrap iterations')
    parser.add_argument('--save-countsmap', action='store_true',
                        help='Save counts map')
    parser.add_argument('--mcbg', action='store_true',
                        help='Use MC background')
    parser.add_argument('--save-cov', action='store_true',
                        help='Save covariance matrix')
    parser.add_argument('--recompute-model', action='store_true',
                        help='Recompute model')
    parser.add_argument('--pnt-src', nargs=4, type=float,
                        help='Add a point source at RA Dec with a flux and spectrum index')
    parser.add_argument('--pnt-srcs', type=str,
                        help='CSV file containing point source information')
    parser.add_argument('--path-sig', type=str, default='',
                        help='Path to MC signal file if not using default')
    parser.add_argument('--isotropic', type=float, default=0,
                        help='Add an isotropic component by fraction of measured')
    parser.add_argument('--inj-template', default='')

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
        recompute_model=args.recompute_model,
        path_sig=args.path_sig)

    if args.pnt_src:
        ra, dec, flux_pt, gamma = args.pnt_src
        pnt_trial_runner = point_source_trial_runner(
            ra, dec, gamma, llh.event_generator.ana)
        ninj_pt = int(pnt_trial_runner.to_ns(flux_pt, E0=1, unit=1e3))

    if args.pnt_srcs:
        pnt_srcs = pd.read_csv(args.pnt_srcs)
        pnts_trial_runners = []
        ninj_pts = []
        for idx, row in pnt_srcs.iterrows():
            ra, dec, flux_pt = row['RA'], row['Dec'], row['Flux']
            pnts_trial_runners.append(point_source_trial_runner(
                ra, dec, 2.5, llh.event_generator.ana))
            ninj_pts.append(int(pnts_trial_runners[-1].to_ns(flux_pt, E0=1, unit=1e3)))

    if args.isotropic:
        isotropic_galaxy_sample = GALAXY_LIBRARY.get_sample('flat')
        isotropic_event_generator = CskyEventGenerator(
            args.N_yr,
            isotropic_galaxy_sample,
            gamma=2.28,
            Ebinmin=args.ebinmin,
            Ebinmax=args.ebinmax,
            idx_mask=llh.idx_mask,
            mc_background=args.mcbg,
            path_sig=args.path_sig
        )
        isotropic_counts = isotropic_event_generator.trial_runner.to_ns(
            (1e5)**2 * 1.44e-18 * llh.f_sky * 4 * np.pi, E0=1e5, unit=1)
        
    if args.inj_template:
        inj_galaxy_sample = GALAXY_LIBRARY.get_sample(args.inj_template)
        inj_event_generator = CskyEventGenerator(
            args.N_yr,
            inj_galaxy_sample,
            gamma=2.28,
            Ebinmin=args.ebinmin,
            Ebinmax=args.ebinmax,
            idx_mask=llh.idx_mask,
            mc_background=args.mcbg,
            path_sig=args.path_sig
        )

    result_list = []

    for ninj in args.inject:
        for i in tqdm(np.arange(args.n_trials)):
            if args.inj_template:
                trial = inj_event_generator.SyntheticTrial(
                    ninj, keep_total_constant=False)[0]
            else:
                trial = llh.event_generator.SyntheticTrial(
                    ninj, keep_total_constant=False)[0]
            if args.pnt_src:
                pt_trial = pnt_trial_runner.get_one_trial(ninj_pt)[0]
                trial = append_signal_trials(trial, pt_trial)
            if args.pnt_srcs:
                for n, ptr in zip(ninj_pts, pnts_trial_runners):
                    pts_trial = ptr.get_one_trial(n)[0]
                    trial = append_signal_trials(trial, pts_trial)
            if args.isotropic:
                isotropic_trial = isotropic_event_generator.SyntheticTrial(
                    int(args.isotropic * isotropic_counts))[0]
                trial = append_signal_trials(trial, isotropic_trial)
            ns = NeutrinoSample()
            ns.inputTrial(trial, ana=llh.event_generator.ana)
            if llh.acceptance is None:
                llh.acceptance = ns.calc_effective_area(llh.event_generator.ana)
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
            result_dict['f_inj'] = calc_f_inj(llh, ninj, trial)
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
