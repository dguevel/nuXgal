import argparse
import json
from copy import deepcopy

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
    parser.add_argument('--compute-std', action='store_true')
    parser.add_argument('--true-galaxies', action='store_true')
    #parser.add_argument('--analysis', nargs='+', default=['weighted'])
    args = parser.parse_args()
    print(args.n_trials, args.output)

    #weighted_llh = WeightedLikelihood(10, 'WISE', args.compute_std, 0, 1, 50, gamma=args.gamma)
    unweighted_llh = Likelihood('v4', 'WISE', args.compute_std, 0, 3, 1, gamma=args.gamma)

    if args.true_galaxies:
        true_template = hp.read_map(Defaults.GALAXYMAP_TRUE_FORMAT.format(galaxyName='WISE'))
        conf = {
            'ana': unweighted_llh.event_generator.ana,
            'template': true_template,
            'flux': cy.hyp.PowerLawFlux(args.gamma),
            'fitter_args': dict(gamma=args.gamma),
            'sigsub': True,
            'fast_weight': True,
        }
        trial_runner = cy.get_trial_runner(conf)

    else:
        trial_runner = unweighted_llh.event_generator.trial_runner
    #eg = weighted_llh.event_generator
    eg = unweighted_llh.event_generator
    #eg.updateGamma(args.gamma)

    result_list = []
    for n_inject in args.n_inject:
        for i in range(args.n_trials):
            results = {}
            trial, nexc = trial_runner.get_one_trial(n_inject)
            results['dof'] = Defaults.MAX_L-unweighted_llh.lmin
            results['n_inj'] = n_inject
            results['flux_inj'] = trial_runner.to_dNdE(n_inject, E0=1e5) / (4*np.pi*unweighted_llh.f_sky)
            results['gamma'] = args.gamma
            results['n_inj_i'], results['n_atm_i'] = find_n_inj(trial)

            #weighted_results = weighted_analysis(weighted_llh, trial, args.gamma)
            #for key in weighted_results:
            #    results[key][counter] = weighted_results[key]

            unweighted_results = unweighted_analysis(unweighted_llh, trial, args.gamma)
            unweighted_results['flux_fit'] = eg.trial_runner.to_dNdE(template_results['n_fit'], E0=1e5) / (4*np.pi*unweighted_llh.f_sky)
            for key in unweighted_results:
                results[key] = unweighted_results[key]

            template_results = template_analysis(trial, nexc, eg.trial_runner)
            template_results['template_flux_fit'] = eg.trial_runner.to_dNdE(template_results['template_n_fit'], E0=1e5) / (4*np.pi*unweighted_llh.f_sky)
            for key in template_results:
                results[key] = template_results[key]

            results['template_TS_i'] = []
            results['template_n_fit_i'] = []
            results['template_gamma_i'] = []
            for ebin in range(Defaults.NEbin):
                elo, ehi = Defaults.map_logE_edge[ebin], Defaults.map_logE_edge[ebin]
                ebin_trial = filter_trial(trial, elo, ehi)
                template_results = template_analysis(ebin_trial, nexc, trial_runner)
                results['template_TS_i'].append(template_results['template_TS'])
                results['template_n_fit_i'].append(template_results['template_n_fit'])
                results['template_gamma_i'].append(template_results['template_gamma'])

            result_list.append(results)

    with open(args.output, 'w') as fp:
        json.dump(result_list, fp)

    #data = pd.DataFrame(results)
    #data.to_csv(args.output, index=False)

def weighted_analysis(llh, trial, gamma):
    ns = WeightedNeutrinoSample()
    ns.inputTrial(trial, '10')
    llh.inputData(ns)
    ns.updateCountsMap(gamma, llh.event_generator.ana)
    ns.updateMask(llh.idx_mask)
    result_dict = {}
    result_dict['weighted_f_fit'], result_dict['weighted_TS'] = llh.minimize__lnL()
    deweighted_f_fit = llh.weighted_f_to_f(result_dict['weighted_f_fit'], gamma)
    result_dict['deweighted_f_fit'] = deweighted_f_fit
    result_dict['weighted_n_fit'] = ns.getEventCounts() * deweighted_f_fit
    result_dict['weighted_flux_fit'] = llh.event_generator.trial_runner.to_dNdE(result_dict['weighted_n_fit'], E0=1e5) / (4*np.pi*llh.f_sky)
    result_dict['weighted_chi2'] = -2*llh.log_likelihood([result_dict['weighted_f_fit']], gamma=gamma)
    return result_dict

def find_n_inj(trial):
    n_inj = []
    n_atm = []
    for i in range(Defaults.NEbin):
        elo = Defaults.map_logE_edge[i]
        ehi = Defaults.map_logE_edge[i + 1]
        n_inj.append(0)
        n_atm.append(0)

        for yr in trial:
            if len(yr) > 1:
                idx = (yr[1]['log10energy'] > elo) * (yr[1]['log10energy'] < ehi)
                n_inj[-1] += idx.sum()

            idx = (yr[0]['log10energy'] > elo) * (yr[0]['log10energy'] < ehi)
            n_atm[-1] += idx.sum()
    return n_inj, n_atm


def filter_trial(trial, elo, ehi):
    new_trial = deepcopy(trial)
    for yr in new_trial:
        for evt in yr:
            evt = evt[(evt['log10energy'] > elo) * (evt['log10energy'] < ehi)]

    return new_trial
            


def unweighted_analysis(llh, trial, gamma):
    ns = NeutrinoSample()
    ns.inputTrial(trial, 'v4')
    ns.updateMask(llh.idx_mask)
    llh.inputData(ns)
    result_dict = {}
    #f_fit, result_dict['TS'] = llh.minimize__lnL()
    #result_dict['f_fit'] = np.sum(f_fit * llh.Ncount[llh.Ebinmin:llh.Ebinmax]) / llh.Ncount[llh.Ebinmin:llh.Ebinmax].sum()
    #result_dict['n_fit'] = np.sum(result_dict['f_fit'] * llh.Ncount[llh.Ebinmin:llh.Ebinmax])
    #result_dict['flux_fit'] = llh.event_generator.trial_runner.to_dNdE(result_dict['n_fit'], E0=1e5) / (4*np.pi*llh.f_sky)
    #result_dict['chi2'] = -2*llh.log_likelihood([result_dict['f_fit']], gamma=gamma)

    f_fit, result_dict['TS'] = llh.minimize__lnL()
    result_dict['f_fit'] = list(f_fit)
    result_dict['TS_i'] = [2*(llh.log_likelihood_Ebin(result_dict['f_fit'][i-llh.Ebinmin], i)-llh.log_likelihood_Ebin(0, i)) for i in range(llh.Ebinmin, llh.Ebinmax)]
    result_dict['n_fit'] = list(result_dict['f_fit'] * llh.Ncount[llh.Ebinmin:llh.Ebinmax])

    #result_dict['f_fit'], result_dict['TS'] = llh.minimize__lnL_one_dof()
    #result_dict['n_fit'] = result_dict['f_fit'] * llh.Ncount.sum()
    #result_dict['flux_fit'] = llh.event_generator.trial_runner.to_dNdE(result_dict['n_fit'], E0=1e5) / (4*np.pi*llh.f_sky)
    #result_dict['chi2'] = -2*llh.log_likelihood_one_dof([result_dict['f_fit']])
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
