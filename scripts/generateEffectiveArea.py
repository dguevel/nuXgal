"""Generate IceCube exposure matrix for the default dataspec.

The exposure matrix is defined as the sum(live time * effective area) summed over seasons."""

import argparse

import csky as cy
import healpy as hp
import numpy as np

from KIPAC.nuXgal.DataSpec import data_spec_factory
from KIPAC.nuXgal.GalaxySample import GALAXY_LIBRARY
from KIPAC.nuXgal import Defaults

def main():
    parser = argparse.ArgumentParser('Generate IceCube exposure matrix')
    parser.add_argument('--nyr', default='10')
    args = parser.parse_args()

    data_specs = data_spec_factory(0, 4)
    dataspec = {
        3: data_specs.ps_3yr,
        10: data_specs.ps_10yr,
        'v4': data_specs.ps_v4,
        'ps_v4': data_specs.ps_v4,
        'estes_10yr': data_specs.estes_10yr,
        'dnn_cascade_10yr': data_specs.dnn_cascade_10yr,
        'nt_v5': data_specs.nt_v5}[args.nyr]

    version = {
        'v4': 'version-004-p02',
        'ps_v4': 'version-004-p02',
        'estes_10yr': 'version-001-p03',
        'dnn_cascade_10yr': 'version-001-p01',
        'nt_v5': 'version-005-p01'}[args.nyr]

    # set up csky analysis which does all the bookkeeping
    ana = cy.get_analysis(cy.selections.repo, version, dataspec)

    # for each sub-analysis calculate the exposure map
    exposuremap = np.zeros((Defaults.logE_microbin_edge.size - 1, Defaults.sindec_bin_edge.size - 1))
    for subana in ana:
        exposuremap += effective_area_2d(subana) * subana.livetime

    # save exposure matrix
    with open(Defaults.EXPOSUREMAP_FORMAT.format(year=args.nyr + 'yr'), 'wb') as fp:
        np.save(fp, exposuremap)


def effective_area_2d(a):

    '''
    Adapted from effective area tutorial in csky docs. Returns effective area in m^2.

    Note the following:

    *dlogE is the logarithmic energy bin width

    *solid_angle represents the portion of the sky that we are observing with the detector

    *In the area eqn defined below, 1/ (1e4*np.log(10)) comes from eqn 5.12 in the thesis linked at the start
    of this notebook. The 1e4 accounts for converting from cm^2 -> m^2 and np.log(10) accounts for the ln(10) when
    differentiating dlog(E)
    '''

    logE_bins = Defaults.logE_microbin_edge
    sindec_bins = Defaults.sindec_bin_edge
    dlogE = Defaults.dlogE_micro
    dsindec = Defaults.dsindec
    
    solid_angle=2*np.pi*(np.sin(a.sig.true_dec+dsindec/2)-np.sin(a.sig.true_dec-dsindec/2))
    area= 1/ (1e-4 * np.log(10)) * (a.sig.oneweight / (a.sig.true_energy * solid_angle * dlogE))

    h, _, _ = np.histogram2d(np.log10(a.sig.true_energy), np.sin(a.sig.true_dec), weights=area, bins=(logE_bins, sindec_bins))

    return h

if __name__ == '__main__':
    main()