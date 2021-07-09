"""Top level class to generate synthetic events"""

import os
import numpy as np

import healpy as hp
import csky as cy

from . import Defaults
from . import file_utils
from .Generator import AtmGenerator, AstroGenerator_v2, get_dnde_astro
from .Exposure import ICECUBE_EXPOSURE_LIBRARY


class CskyEventGenerator():
    """Class to generate synthetic IceCube events

    This can generate both atmospheric and astrophysical events
    """
    def __init__(self, year=cy.selections.PSDataSpecs.IC79, astroModel=None, version='version-002-p03', f_sky=1., smeared=True):
        """C'tor
        """

        self.year = year
        self.f_sky = f_sky

        #self.f_astro_north_truth = np.array([0, 0.00221405, 0.01216614, 0.15222642, 0., 0., 0.])# * 2.
        self.f_astro_north_truth = Defaults.f_astro_north_truth.copy()

        ana = cy.get_analysis(cy.selections.repo, version, year)
        self.ana = ana

        cy.CONF['ana'] = ana
        self.ana_dir = Defaults.NUXGAL_ANA_DIR

        galaxy_map = hp.read_map(Defaults.GALAXYMAP_FORMAT.format(galaxyName='WISE'))
        galaxy_map[Defaults.idx_muon] = 0.
        galaxy_map /= galaxy_map.sum()
        self.density_nu = galaxy_map.copy()
        self.nside = hp.npix2nside(galaxy_map.size)


        self.nevts = []

        for emin, emax in zip(Defaults.map_logE_edge, Defaults.map_logE_edge[1:]):
            # calculate number of events in the northern sky
            nevt = 0
            for k in ana:
                idx_mask = (k.bg_data['log10energy'] > emin) & (k.bg_data['log10energy'] < emax) & (k.bg_data['dec'] > (np.pi/2 - Defaults.theta_north))
                nevt += idx_mask.sum()
            self.nevts.append(nevt)

        self.nevts = np.array(self.nevts)

        self.conf = {
            'template': self.density_nu,
            'flux': cy.hyp.PowerLawFlux(gamma=2.28),
            'sigsub': True,
            'fast_weight': False,
            'dir': cy.utils.ensure_dir('{}/templates/WISE'.format(self.ana_dir))
        }

        self.smeared = smeared

        self.trial_runner = cy.get_trial_runner(self.conf)


    def SyntheticData(self, N_yr, f_diff):
        """Generate Synthetic Data

        Parameters
        ----------
        N_yr : `float`
            Number of years of data to generate
        f_diff : `float`
            Fraction of astro events w.r.t. diffuse muon neutrino flux,
            f_diff = 1 means injecting astro events that sum up to 100% of diffuse muon neutrino flux

        Returns
        -----
        counts_map : `np.ndarray`
            Maps of simulated events
        """


        sr = self.f_sky * 4 * np.pi
        #n_inj = (f_diff * Defaults.f_astro_north_truth * self.nevts).sum() # probably change to the trial_runner.to_ns method
        n_inj = int(f_diff * self.conf['flux'].to_ns(1.44e-18 * sr, self.trial_runner.sig_inj_acc_total, E0=100000, E2dNdE=False))

        trial, _ = self.trial_runner.get_one_trial(n_inj)
        print(trial)


        # atmospheric background events
        atm = cy.utils.Arrays.concatenate([t[0] for t in trial])

        # astrophysical signal events
        if n_inj > 0:
            astro = cy.utils.Arrays.concatenate([t[1] for t in trial])

        # N_astro / N_total, populated in energy loop
        f_astro = np.zeros(Defaults.f_astro_north_truth.size)

        # initial sky maps; (n energy bins, healpy pixels)
        astro_maps = np.zeros((Defaults.NEbin, hp.nside2npix(self.nside)))
        atm_maps = np.zeros((Defaults.NEbin, hp.nside2npix(self.nside)))

        for i, (emin, emax) in enumerate(zip(Defaults.map_logE_edge, Defaults.map_logE_edge[1:])):
            if n_inj > 0:
                # select atm events in energy bin and northern hemispher
                atm_mask = (atm['log10energy'] > emin) & (atm['log10energy'] < emax) & (atm['dec'] > (np.pi/2 - Defaults.theta_north)) 

                # select astro events in energy bin; template takes care of dec
                if self.smeared:
                    astro_mask = (astro['log10energy'] > emin) & (astro['log10energy'] < emax) & (astro['dec'] > (np.pi/2 - Defaults.theta_north))
                else:
                    astro_mask = (astro['log10energy'] > emin) & (astro['log10energy'] < emax) & (astro['true_dec'] > (np.pi/2 - Defaults.theta_north))

                # populate f_astro; edge cases in if and elif blocks
                if self.nevts[i] == 0:
                    # happens for highest energy bins, avoid divide by 0
                    f_astro[i] = 1.
                elif astro_mask.sum() > self.nevts[i]:
                    # happens occasionally in the bins with few atm counts
                    f_astro[i] = 1.
                else:
                    f_astro[i] = astro_mask.sum() / self.nevts[i]

                # down select the proper number of atm events
                n_atm = int(self.nevts[i] * (1 - f_astro[i]))
                atm_subset = atm[atm_mask]
                included_events = np.random.choice(len(atm_subset), size=n_atm, replace=False)
                atm_subset = atm_subset[included_events]

                # create sky maps
                atm_maps[i] = event2map(atm_subset['ra'], atm_subset['dec'], self.nside)
                if self.smeared:
                    astro_maps[i] = event2map(astro['ra'][astro_mask], astro['dec'][astro_mask], self.nside)
                else:
                    astro_maps[i] = event2map(astro['true_ra'][astro_mask], astro['true_dec'][astro_mask], self.nside)

            else:
                # create sky maps
                atm_mask = (atm['log10energy'] > emin) & (atm['log10energy'] < emax) & (atm['dec'] > (np.pi/2 - Defaults.theta_north)) 
                atm_maps[i] = event2map(atm['ra'][atm_mask], atm['dec'][atm_mask], self.nside)
                astro_maps[i] = np.zeros(hp.nside2npix(self.nside))

        self.f_astro_inj = f_astro.copy()
        data_maps = astro_maps + atm_maps
        return data_maps


# TODO: check if this is implemented elsewhere
def event2map(ra, dec, nside=128):
    npix = hp.nside2npix(nside)
    theta = np.pi/2 - dec
    trial_map = np.histogram(hp.ang2pix(nside, theta, ra), bins=np.arange(npix + 1))[0]
    return trial_map
