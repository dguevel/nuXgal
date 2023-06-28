import argparse

import numpy as np
from tqdm import tqdm

from KIPAC.nuXgal import Defaults
from KIPAC.nuXgal.Likelihood import Likelihood
from KIPAC.nuXgal.NeutrinoSample import NeutrinoSample
from KIPAC.nuXgal.DataSpec import ps_3yr, ps_10yr, ps_v4, estes_10yr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n-inject', help='Number of signal nu to inject', type=int, default=0)
    parser.add_argument('-r', '--repititions', help='Number of repetitions', type=int, default=100)
    parser.add_argument('-y', '--year', help='Csky data to use', type=str, default='v4')
    parser.add_argument('-o', '--output', help='Output filename', type=str)
    parser.add_argument('--no-background', action='store_true', help='Do not include background events in signal simulation. Using this option will overrule -n')
    parser.add_argument('--ebinmin', default=0, type=int)
    parser.add_argument('--ebinmax', default=3, type=int)
    args = parser.parse_args()

    # set up likelihood
    llh = Likelihood(args.year, 'WISE', False, args.ebinmin, args.ebinmax, 1, 2.5)
    ns = NeutrinoSample()

    # run iterations
    w_data = np.zeros((args.repititions, Defaults.NEbin, Defaults.MAX_L + 1))
    for i in tqdm(range(args.repititions)):
        if args.no_background:
            n_inject = llh.Ncount_atm.sum()
            trial, nexc = llh.event_generator.SyntheticTrial(args.n_inject, llh.idx_mask, signal_only=True)
            for tr in trial:
                if len(tr) == 2:
                    tr.pop(0)
        else:
            trial, nexc = llh.event_generator.SyntheticTrial(args.n_inject, llh.idx_mask)
        ns.inputTrial(trial, args.year)
        ns.updateMask(llh.idx_mask)
        llh.inputData(ns)
        w_data[i] = llh.w_data.copy()

    # calculate std
    w_std = np.std(w_data, axis=0)

    # write std
    np.save(args.output, w_std)


if __name__ == '__main__':
    main()