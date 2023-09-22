import argparse
import json
from tqdm import tqdm

import numpy as np

from KIPAC.nuXgal.NeutrinoSample import NeutrinoSample
from KIPAC.nuXgal.Likelihood import Likelihood
from KIPAC.nuXgal.BeamLikelihood import BeamLikelihood
from KIPAC.nuXgal import Defaults


def main():
    """Main function for executing the script."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n-trials', help='Number of trials', type=int)
    parser.add_argument('-i', '--n-inject',
                        help='Number of neutrinos to inject',
                        type=int, nargs='+')
    parser.add_argument('-o', '--output')
    parser.add_argument('--gamma', help='Injection spectrum power law index',
                        default=2.5, type=float)
    parser.add_argument('--galaxy-catalog',
                        help='Galaxy catalog to cross correlate',
                        choices=['WISE', 'Planck', 'unWISE_z=0.4'],
                        default='unWISE_z=0.4')
    parser.add_argument('--nyear',
                        default='v4',
                        help='PS Tracks version')
    parser.add_argument('--compute-std', action='store_true')
    parser.add_argument('--ebinmin',
                        default=0,
                        type=int,
                        help='Minimum energy bin')
    parser.add_argument('--ebinmax',
                        default=3,
                        type=int,
                        help='Maximum energy bin')
    parser.add_argument('--lmin',
                        default=50, 
                        type=int,
                        help='Minimum multipole')
    parser.add_argument('--save-cls',
                        action='store_true',
                        help='Save cross spectrum')
    parser.add_argument('--bootstrap-niter', default=100, type=int)
    parser.add_argument('--err-type', default='bootstrap',
                        help='Error calculation method',
                        choices=['bootstrap', 'polspice'])
    parser.add_argument('--lbin',
                        default=4,
                        type=int,
                        help='Cross spectrum bin width')
    parser.add_argument('--unblind',
                        action='store_true',
                        help='Unblind the analysis')
    args = parser.parse_args()

    llh = BeamLikelihood(
        N_yr=args.nyear,
        galaxyName=args.galaxy_catalog,
        recompute_model=args.compute_std,
        Ebinmin=args.ebinmin,
        Ebinmax=args.ebinmax,
        lmin=args.lmin,
        gamma=args.gamma,
        err_type=args.err_type,
        lbin=args.lbin)

    trial_runner = llh.event_generator.trial_runner
    eg = llh.event_generator

    result_list = []


    if args.unblind:
        trial, nexc = llh.event_generator.trial_runner.get_one_trial(TRUTH=True)
        results = {}
        results['dof'] = Defaults.MAX_L-llh.lmin-1
        results['n_inj'] = n_inject
        results['flux_inj'] = trial_runner.to_dNdE(n_inject, E0=1e5, gamma=2.5) / (4*np.pi*llh.f_sky)
        results['gamma'] = args.gamma
        results['ebinmin'] = args.ebinmin
        results['ebinmax'] = args.ebinmax
        results['logemin'] = Defaults.map_logE_edge[args.ebinmin]
        results['logemax'] = Defaults.map_logE_edge[args.ebinmax]
        results['galaxy_catalog'] = args.galaxy_catalog
        results['lmin'] = args.lmin
        results['N_yr'] = args.nyear
        results['bootstrap_niter'] = args.bootstrap_niter
        results['err-type'] = args.err_type
        results['lbin'] = args.lbin

        crosscorr_results = crosscorr_analysis(llh, trial, args)
        crosscorr_results['flux_fit'] = eg.trial_runner.to_dNdE(np.sum(crosscorr_results['n_fit']), E0=1e5, gamma=2.5) / (4*np.pi*llh.f_sky)
        for key in crosscorr_results:
            results[key] = crosscorr_results[key]

        results['n_total'] = int(np.sum(llh.Ncount))
        results['n_total_i'] = dict(zip(range(args.ebinmin, args.ebinmax), llh.Ncount[args.ebinmin:args.ebinmax].astype(int).tolist()))
        results['n_inj_i'] = dict(zip(range(args.ebinmin, args.ebinmax), find_n_inj_per_bin(trial, args.ebinmin, args.ebinmax)))
        results['f_inj_i'] = dict(zip(range(args.ebinmin, args.ebinmax), (find_n_inj_per_bin(trial, args.ebinmin, args.ebinmax)/llh.Ncount[args.ebinmin:args.ebinmax]).tolist()))
        results['f_inj'] = float(results['n_inj'] / results['n_total'])

        result_list.append(results)

    else:
        for n_inject in args.n_inject:
            for i in tqdm(range(args.n_trials)):
                results = {}
                trial, nexc = llh.event_generator.SyntheticTrial(n_inject, keep_total_constant=False)
                results['dof'] = Defaults.MAX_L-llh.lmin-1
                results['n_inj'] = n_inject
                results['flux_inj'] = trial_runner.to_dNdE(n_inject, E0=1e5, gamma=2.5) / (4*np.pi*llh.f_sky)
                results['gamma'] = args.gamma
                results['ebinmin'] = args.ebinmin
                results['ebinmax'] = args.ebinmax
                results['logemin'] = Defaults.map_logE_edge[args.ebinmin]
                results['logemax'] = Defaults.map_logE_edge[args.ebinmax]
                results['galaxy_catalog'] = args.galaxy_catalog
                results['lmin'] = args.lmin
                results['N_yr'] = args.nyear
                results['bootstrap_niter'] = args.bootstrap_niter
                results['err-type'] = args.err_type
                results['lbin'] = args.lbin

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


def crosscorr_analysis(llh, trial, args):
    """
    Perform cross-correlation analysis.

    Parameters:
        llh (Likelihood): The Likelihood instance.
        trial (list): The trial data.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        dict: The cross-correlation analysis results.
    """
    ns = NeutrinoSample()
    ns.inputTrial(trial)
    ns.updateMask(llh.idx_mask)
    llh.inputData(ns, bootstrap_niter=args.bootstrap_niter)
    result_dict = {}

    f_fit, result_dict['TS'] = llh.minimize__lnL()
    result_dict['f_fit'] = list(f_fit)
    result_dict['TS_i'] = [2*(llh.log_likelihood_Ebin(result_dict['f_fit'][i-llh.Ebinmin], i)-llh.log_likelihood_Ebin(0, i)) for i in range(llh.Ebinmin, llh.Ebinmax)]
    result_dict['n_fit'] = list(result_dict['f_fit'] * llh.Ncount[llh.Ebinmin:llh.Ebinmax])
    result_dict['chi_square'] = [-2*float(llh.log_likelihood_Ebin(f_fit[i-llh.Ebinmin], i)) for i in range(llh.Ebinmin, llh.Ebinmax)]

    if args.save_cls:
        result_dict['cls'] = {}
        result_dict['cls_std'] = {}
        for ebin in range(args.ebinmin, args.ebinmax):
            result_dict['cls'][ebin] = llh.w_data[ebin].tolist()
            result_dict['cls_std'][ebin] = llh.w_std[ebin].tolist()

    return result_dict


def template_analysis(trial, nexc, trial_runner):
    """
    Perform template analysis.

    Parameters:
        trial (list): The trial data.
        nexc (int): Number of excess events.
        trial_runner: The trial runner instance.

    Returns:
        dict: The template analysis results.
    """
    result = {}
    fit_res = trial_runner.get_one_fit_from_trial((trial, nexc))
    if len(fit_res) == 2:
        result['template_TS'], result['template_n_fit'] = fit_res
    elif len(fit_res) == 3:
        result['template_TS'], result['template_n_fit'], result['template_gamma'] = fit_res
    return result


if __name__ == '__main__':
    main()
