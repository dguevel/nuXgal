"""Create a beam file for a given neutrino selection assuming
azimuthal symmetry."""

import argparse
import os

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from KIPAC.nuXgal import Defaults
from KIPAC.nuXgal.CskyEventGenerator import CskyEventGenerator
from KIPAC.nuXgal.GalaxySample import GALAXY_LIBRARY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'nu_selection',
        help='Neutrino selection to use',
        choices=['ps_v4', 'estes_10yr', 'dnn_cascade_10yr', 'nt_v5']
    )
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save debugging plots'
    )

    args = parser.parse_args()

    # galaxy sample doesn't matter, just need to get the right data spec
    galaxy_sample = GALAXY_LIBRARY.get_sample('unWISE_z=0.4')
    event_generator = CskyEventGenerator(
        args.nu_selection,
        galaxy_sample,
        gamma=2.5,
        Ebinmin=0,
        Ebinmax=-1,
        idx_mask=Defaults.idx_muon)

    # iterate over energy bins
    beams = []
    for ebin, (elo, ehi) in enumerate(zip(Defaults.map_logE_edge, Defaults.map_logE_edge[1:])):

        probs = event_generator.trial_runner.sig_inj_probs
        psf = 0
        for i, subana in enumerate(event_generator.ana):

            delta_dec = subana.sig['dec'] - subana.sig['true_dec']
            delta_ra = subana.sig['ra'] - subana.sig['true_ra']
            dtheta = np.sqrt(
                delta_ra**2 * np.cos(subana.sig['true_dec'])**2 + delta_dec**2
            )
            weight = subana.sig['sindec'] > np.sin(-5*np.pi/180)
            weight = weight.astype(float)
            weight *= subana.sig['oneweight']
            weight *= (subana.sig['energy'] > 10**elo)
            weight *= (subana.sig['energy'] < 10**ehi)
            weight /= dtheta

            p, bins = np.histogram(
                dtheta,
                bins=np.linspace(0, .1, 50),
                weights=weight
            )
            psf += p * probs[i]
            angles = (bins[1:] + bins[:-1]) / 2
        beam = hp.beam2bl(psf, theta=angles, lmax=Defaults.MAX_L) * probs[i]
        beam /= beam[0]
        beams.append(beam)

        if args.save_plots:
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax = ax.flatten()
            ax[0].plot(np.degrees(angles), psf)
            ax[0].set_xlabel('Angular distance (deg)')
            ax[0].set_ylabel('PSF')
            ax[0].set_title('{nu} {elo:.2f} < log10(E/GeV) < {ehi:.2f}'.format(
                nu=args.nu_selection, elo=elo, ehi=ehi))

            ax[1].plot(beam)
            ax[1].set_xlabel('Multipole')
            ax[1].set_ylabel('Beam function')

            plot_fname = 'beam_{nu}_elo_{elo:.2f}_ehi_{ehi:.2f}.png'
            plot_fname = plot_fname.format(
                nu=args.nu_selection,
                elo=elo,
                ehi=ehi
                )
            plot_fname = os.path.join(Defaults.NUXGAL_PLOT_DIR, plot_fname)

            plt.savefig(plot_fname, bbox_inches='tight')
            plt.savefig(
                plot_fname.replace('.png', '.pdf'), bbox_inches='tight')

        fname = Defaults.BEAM_FNAME_FORMAT.format(
            year=args.nu_selection, ebin=ebin)
        np.save(fname, beam)

if __name__ == "__main__":
    main()
