#!/usr/bin/env python 

"""Calculate TS distribution for the template analysis"""

from argparse import ArgumentParser

import healpy as hp
import numpy as np
import csky as cy

from KIPAC.nuXgal.Likelihood import Likelihood

def main():
    parser = ArgumentParser(__doc__)
    parser.add_argument('-n', type=int, default=1,
                        help='Number of trials')
    parser.add_argument('-i', type=int, default=0,
                        help='Number of events to inject')
    parser.add_argument('-o', '--output', type=str, default='template-trials.npy',
                        help='Output file')
    parser.add_argument('--gamma', type=float, default=2.0,
                        help='Spectral index')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    llh = Likelihood('nt_v5', 'unWISE_z=0.4', 0, -1, 10, args.gamma, False, True)

    fits = llh.event_generator.trial_runner.get_many_fits(args.n, args.i)
    data = np.array([fits['ns'], fits['gamma'], fits['ts']]).T
    np.save(args.output, data)

if __name__ == '__main__':
    main()
