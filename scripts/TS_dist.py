import argparse

import numpy as np
import pandas as pd

from KIPAC.nuXgal.CskyEventGenerator import CskyEventGenerator
from KIPAC.nuXgal.NeutrinoSample import NeutrinoSample
from KIPAC.nuXgal.Likelihood import Likelihood

import matplotlib.pyplot as plt
import healpy as hp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n-trials', help='Number of trials', type=int)
    parser.add_argument('-i', '--n-inject', help='Number of neutrinos to inject', type=int)
    parser.add_argument('-o', '--output')
    args = parser.parse_args()
    print(args.n_trials, args.output)

    f = np.zeros(args.n_trials)
    TS = np.zeros(args.n_trials)

    llh = Likelihood(10, 'WISE', False, 0, 1, 50)
    eg = llh.event_generator

    for i in range(args.n_trials):
        countsmap, ninj = eg.SyntheticData(args.n_inject)
        ns = NeutrinoSample()
        ns.inputCountsmap(countsmap)
        llh.inputData(ns)
        f[i], TS[i] = llh.minimize__lnL()

    data = pd.DataFrame({'f': f, 'TS': TS})
    data.to_csv(args.output, index=False)

if __name__ == '__main__':
    main()