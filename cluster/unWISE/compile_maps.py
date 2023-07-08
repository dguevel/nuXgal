#!/usr/bin/env python3

from argparse import ArgumentParser
import os

import numpy as np
import healpy as hp

def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', nargs='+', help='Map files')
    parser.add_argument('-o', '--output', help='Output directory')
    parser.add_argument('--nside', default=128, type=int)
    args = parser.parse_args()

    nside = args.nside
    npix = hp.nside2npix(nside)
    template = np.zeros((4, npix))

    for filename in args.input:
        template += np.load(filename)[:4,]

    datapath = '/data/user/dguevel/Fermi-LAT_pi0_map.npy'
    mask = np.load(datapath)

    mask_bool = mask.copy()
    thresh = np.percentile(mask, 75)
    mask_bool[mask < thresh] = 1.
    mask_bool[mask >= thresh] = 0.
    mask_bool = mask_bool.astype(bool)

    z = [0.4, 0.6, 1.0, 1.5]
    for i in range(4):

        # save galaxy map
        hp.write_map(os.path.join(args.output, 'unWISE_z={0:1.1f}_galaxymap.fits'.format(z[i])), template[i], overwrite=True)

        # save overdensity
        overdensity = template[i] / template[i].mean() - 1.
        hp.write_map(os.path.join(args.output, 'unWISE_z={0:1.1f}_overdensity.fits'.format(z[i])), overdensity, overwrite=True)

        # save overdensity alm
        alm = hp.map2alm(overdensity)
        hp.write_alm(os.path.join(args.output, 'unWISE_z={0:1.1f}_overdensityalm.fits'.format(z[i])), alm, overwrite=True)

        # save galactic plane mask
        hp.write_map(os.path.join(args.output, 'unWISE_z={0:1.1f}_mask_bool.fits'.format(z[i])), mask_bool, overwrite=True)

if __name__ == '__main__':
    main()