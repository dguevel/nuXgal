#!/usr/bin/env python3

import csky as cy
import healpy as hp
import numpy as np
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument("-n", "--ntrials", default=100, type=int)
    parser.add_argument("-i", "--n-inj", default=0, type=int, nargs='+')
    parser.add_argument("-o", "--output", default="template_trials.npy")
    parser.add_argument("--gamma", default=2.5, type=float)
    parser.add_argument("--freeze-gamma", action="store_true", help="Freeze gamma in the fit")
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()

    template = hp.read_map("/home/dguevel/git/nuXgal/data/ancil/unWISE_z=0.4_galaxymap.fits")

    ana = cy.conf.get_analysis(
        cy.selections.repo,
        "version-004-p02",
        cy.selections.PSDataSpecs.ps_v4,
        #dir=cy.utils.ensure_dir("/data/user/dguevel/template/unWISE_v0.4_PS_v4/ana")
    )
    #ana.save("/data/user/dguevel/template/unWISE_v0.4_PS_v4/ana")

    conf = {
        'ana': ana,
        'template': template,
        'flux': cy.hyp.PowerLawFlux(args.gamma),
        'sigsub': True,
        'fast_weight': True,
        'dir': cy.utils.ensure_dir("/data/user/dguevel/template/unWISE_v0.4_PS_v4/templates")
    }

    if args.freeze_gamma:
        conf['fitter_args']['gamma'] = args.gamma

    trial_runner = cy.get_trial_runner(conf)

    data_arrays = []
    for ninj in args.n_inj:
        f = trial_runner.get_many_fits(args.ntrials, n_sig=ninj, mp_cpus=1)
        f['n_inj'] = np.full_like(f['ts'], ninj)
        data_arrays.append(f)
    fit_results = cy.utils.Arrays.concatenate(data_arrays)
    np.save(args.output, fit_results.as_array)


if __name__ == "__main__":
    main()
