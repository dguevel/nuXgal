"""Unblind the cross correlation analysis"""

import json
from argparse import ArgumentParser

import numpy as np

from KIPAC.nuXgal.Likelihood import Likelihood
from KIPAC.nuXgal.NeutrinoSample import NeutrinoSample
from KIPAC.nuXgal import Defaults


def main():
    parser = ArgumentParser(description=main.__doc__)
    parser.add_argument("-o", "--output", help="Output file name")
    parser.add_argument("--lmin", default=1, type=int, help="Minimum multipole to use")
    parser.add_argument("--galaxy-catalog", default="unWISE_z=0.4", type=str, help="Galaxy sample name")
    parser.add_argument("--ebinmin", type=int, default=0, help="Minimum energy bin")
    parser.add_argument("--ebinmax", type=int, default=Defaults.NEbin, help="Maximum energy bin")
    parser.add_argument("--N-yr", type=str, default="nt_v5", help="Data set name")
    parser.add_argument("--bootstrap-niter", type=int, default=100, help="Number of bootstrap iterations")
    parser.add_argument("--unblind", action="store_true", help="Unblind the analysis")

    args = parser.parse_args()

    llh = Likelihood(
        N_yr=args.N_yr,
        galaxy_catalog=args.galaxy_catalog,
        ebinmin=args.ebinmin,
        ebinmax=args.ebinmax,
        gamma=2.5,
        lmin=args.lmin,
    )

    result_dict = {}
    trial, nexc = llh.event_generator.trial_runner.get_one_trial(truth=args.unblind)
    ns = NeutrinoSample()
    ns.inputTrial(trial)
    llh.inputData(ns, bootstrap_niter=args.bootstrap_niter)
    f_fit, result_dict["TS"] = llh.minimize__lnL()
    result_dict["f_fit"] = f_fit.tolist()
    result_dict['TS_i'] = []
    result_dict['chi_square'] = []
    for i, ebin in enumerate(range(llh.Ebinmin, llh.Ebinmax)):
        result_dict['TS_i'].append(2 * (llh.log_likelihood_Ebin(result_dict['f_fit'][i], ebin) - llh.log_likelihood_Ebin(0, ebin)))
        result_dict['chi_square'].append(llh.chi_square_Ebin(result_dict['f_fit'][i], ebin))
    result_dict['n_fit'] = list(result_dict['f_fit'] * llh.Ncount)
    result_dict['flux_fit'] = llh.event_generator.trial_runner.to_dNdE(sum(result_dict['n_fit']), E0=1e5, gamma=2.5) / (4*np.pi*llh.f_sky)

    with open(args.output, 'w') as f:
        json.dump([result_dict], f, indent=4)



if __name__ == "__main__":
    main()