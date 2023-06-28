import argparse
import json
from tqdm import tqdm

import numpy as np
import pandas as pd

from KIPAC.nuXgal.CskyEventGenerator import CskyEventGenerator
from KIPAC.nuXgal.NeutrinoSample import NeutrinoSample
from KIPAC.nuXgal.WeightedNeutrinoSample import WeightedNeutrinoSample
from KIPAC.nuXgal.WeightedLikelihood import WeightedLikelihood
from KIPAC.nuXgal.Likelihood import Likelihood
from KIPAC.nuXgal.DataSpec import ps_10yr, ps_3yr, estes_10yr
from KIPAC.nuXgal import Defaults

import matplotlib.pyplot as plt
import csky as cy
import healpy as hp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n-trials', help='Number of trials', type=int)
    parser.add_argument('-i', '--n-inject', help='Number of neutrinos to inject', type=int, nargs='+')
    parser.add_argument('-o', '--output')
    parser.add_argument('--gamma', help='Injection spectrum power law index', default=2.5, type=float)
    parser.add_argument('--galaxy-catalog', help='Galaxy catalog to cross correlate', choices=['WISE', 'Planck', 'unWISE_z=0.4'], default='WISE')
    parser.add_argument('--compute-std', action='store_true')
    parser.add_argument('--true-galaxies', action='store_true')
    parser.add_argument('--do-template-bins', action='store_true', help='Do a template analysis fit in individual energy bins.')
    parser.add_argument('--ebinmin', default=0, type=int)
    parser.add_argument('--ebinmax', default=3, type=int)
    parser.add_argument('--lmin', default=50, type=int)
    parser.add_argument('--save-cls', action='store_true')
    parser.add_argument('--bootstrap-error', default=[], nargs='+', type=int)
    parser.add_argument('--century-cube', action='store_true')
    args = parser.parse_args()

    llh = Likelihood('v4', args.galaxy_catalog, args.compute_std, args.ebinmin, args.ebinmax, args.lmin, gamma=args.gamma)

    if args.true_galaxies:
        raise NotImplementedError
        true_template = hp.read_map(Defaults.GALAXYMAP_TRUE_FORMAT.format(galaxyName=args.galaxy_catalog))
        conf = {
            'ana': llh.event_generator.ana,
            'template': true_template,
            'flux': cy.hyp.PowerLawFlux(args.gamma),
            #'fitter_args': dict(gamma=args.gamma),
            'sigsub': True,
            'fast_weight': True,
        }
        trial_runner = cy.get_trial_runner(conf)

    else:
        trial_runner = llh.event_generator.trial_runner
    eg = llh.event_generator

    result_list = []
    for n_inject in args.n_inject:
        for i in tqdm(range(args.n_trials)):
            results = {}
            if args.century_cube:
                trial = []
                for i in range(10):
                    tr, nexc = llh.event_generator.SyntheticTrial(n_inject, keep_total_constant=False)
                    trial.extend(tr)
            else:
                trial, nexc = llh.event_generator.SyntheticTrial(n_inject, keep_total_constant=False)
            results['dof'] = Defaults.MAX_L-llh.lmin-1
            results['n_inj'] = n_inject
            results['flux_inj'] = trial_runner.to_dNdE(n_inject, E0=1e5, gamma=2.5) / (4*np.pi*llh.f_sky)
            results['gamma'] = args.gamma
            results['ebinmin'] = args.ebinmin
            results['ebinmax'] = args.ebinmax
            results['galaxy_catalog'] = args.galaxy_catalog
            results['lmin'] = args.lmin

            crosscorr_results = crosscorr_analysis(llh, trial, args)
            crosscorr_results['flux_fit'] = eg.trial_runner.to_dNdE(np.sum(crosscorr_results['n_fit']), E0=1e5, gamma=2.5) / (4*np.pi*llh.f_sky)
            for key in crosscorr_results:
                results[key] = crosscorr_results[key]

            template_results = template_analysis(trial, nexc, eg.trial_runner)
            template_results['template_flux_fit'] = eg.trial_runner.to_dNdE(template_results['template_n_fit'], E0=1e5, gamma=2.5) / (4*np.pi*llh.f_sky)
            for key in template_results:
                results[key] = template_results[key]

            results['n_total'] = int(np.sum(llh.Ncount))
            results['n_total_i'] = dict(zip(range(args.ebinmin, args.ebinmax), llh.Ncount[args.ebinmin:args.ebinmax].astype(int).tolist()))
            results['n_inj_i'] = dict(zip(range(args.ebinmin, args.ebinmax), find_n_inj_per_bin(trial, args.ebinmin, args.ebinmax)))
            results['f_inj_i'] = dict(zip(range(args.ebinmin, args.ebinmax), (find_n_inj_per_bin(trial, args.ebinmin, args.ebinmax)/llh.Ncount[args.ebinmin:args.ebinmax]).tolist()))
            results['f_inj'] = float(results['n_inj'] / results['n_total'])

            result_list.append(results)

    with open(args.output, 'w') as fp:
        json.dump(result_list, fp, indent=4)


def find_n_inj_per_bin(trial, ebinmin, ebinmax):
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

def crosscorr_analysis(llh, trial, args):
    ns = NeutrinoSample()
    ns.inputTrial(trial, 'v4')
    ns.updateMask(llh.idx_mask)
    llh.inputData(ns, bootstrap_error=args.bootstrap_error)
    result_dict = {}

    f_fit, result_dict['TS'] = llh.minimize__lnL()
    result_dict['f_fit'] = list(f_fit)
    result_dict['TS_i'] = [2*(llh.log_likelihood_Ebin(result_dict['f_fit'][i-llh.Ebinmin], i)-llh.log_likelihood_Ebin(0, i)) for i in range(llh.Ebinmin, llh.Ebinmax)]
    result_dict['n_fit'] = list(result_dict['f_fit'] * llh.Ncount[llh.Ebinmin:llh.Ebinmax])
    result_dict['chi_square'] = [float(llh.chi_square_Ebin(f_fit[i-llh.Ebinmin], i)) for i in range(llh.Ebinmin, llh.Ebinmax)]

    if args.save_cls:
        result_dict['cls'] = {}
        for ebin in range(args.ebinmin, args.ebinmax):

            result_dict['cls'][ebin] = llh.w_data[ebin].tolist()

    return result_dict

def template_analysis(trial, nexc, trial_runner):
    result = {}
    fit_res = trial_runner.get_one_fit_from_trial((trial, nexc))
    if len(fit_res) == 2:
        result['template_TS'], result['template_n_fit'] = fit_res
    elif len(fit_res) == 3:
        result['template_TS'], result['template_n_fit'], result['template_gamma'] = fit_res
    return result


if __name__ == '__main__':
    main()
