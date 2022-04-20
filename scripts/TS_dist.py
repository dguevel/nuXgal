import argparse

import numpy as np
import pandas as pd

from KIPAC.nuXgal.CskyEventGenerator import CskyEventGenerator
from KIPAC.nuXgal.NeutrinoSample import NeutrinoSample
from KIPAC.nuXgal.WeightedNeutrinoSample import WeightedNeutrinoSample
from KIPAC.nuXgal.Likelihood import Likelihood

import matplotlib.pyplot as plt
import healpy as hp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n-trials', help='Number of trials', type=int)
    parser.add_argument('-i', '--n-inject', help='Number of neutrinos to inject', type=int)
    parser.add_argument('-o', '--output')
    parser.add_argument('--gamma', help='Injection spectrum power law index', default=2.5, type=float)
    parser.add_argument('--compute-std', action='store_true')
    args = parser.parse_args()
    print(args.n_trials, args.output)

    weighted_f = np.zeros(args.n_trials)
    gamma = np.zeros(args.n_trials)
    unweighted_f = np.zeros(args.n_trials)
    nfit = np.zeros(args.n_trials)
    flux_fit = np.zeros(args.n_trials)
    flux_inj = np.zeros(args.n_trials)
    TS = np.zeros(args.n_trials)

    llh = Likelihood(10, 'WISE', args.compute_std, 0, 1, 50, gamma=args.gamma)
    llh.getTemplate()
    eg = llh.event_generator

    for i in range(args.n_trials):
        trial = eg.SyntheticTrial(args.n_inject)
        ns = WeightedNeutrinoSample()
        ns.inputTrial(trial)
        llh.inputData(ns)
        ns.updateCountsMap(args.gamma, llh.event_generator.ana)
        weighted_f[i], TS[i] = llh.minimize__lnL()
        unweighted_f[i] = llh.weighted_f_to_f(weighted_f[i], args.gamma)
        nfit[i] = llh.Ncount * weighted_f[i]
        flux_fit[i] = llh.event_generator.trial_runner.to_dNdE(nfit[i], E0=1e5) / (4*np.pi*llh.f_sky)
        flux_inj[i] = llh.event_generator.trial_runner.to_dNdE(args.n_inject, E0=1e5) / (4*np.pi*llh.f_sky)


    data = pd.DataFrame({
        'weighted_f': weighted_f,
        'TS': TS,
        'unweighted_f': unweighted_f,
        'nfit': nfit,
        'ninj': args.n_trials * [args.n_inject,],
        'flux_fit': flux_fit,
        'flux_inj': flux_inj})
    data.to_csv(args.output, index=False)

if __name__ == '__main__':
    main()
