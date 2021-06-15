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
    def __init__(self, year=cy.selections.PSDataSpecs.IC79, astroModel=None, version='version-002-p03'):
        """C'tor
        """

        self.year = year

        #self.f_astro_north_truth = np.array([0, 0.00221405, 0.01216614, 0.15222642, 0., 0., 0.])# * 2.
        self.f_astro_north_truth = Defaults.f_astro_north_truth.copy()

        ana = cy.get_analysis(cy.selections.repo, version, year)
        self.ana = ana

        cy.CONF['ana'] = ana
        ana_dir = Defaults.NUXGAL_ANA_DIR

        galaxy_map = hp.read_map(Defaults.GALAXYMAP_FORMAT.format(galaxyName='WISE'))
        galaxy_map[Defaults.idx_muon] = 0.
        galaxy_map /= galaxy_map.sum()
        self.galaxy_map = galaxy_map.copy()
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

        conf = {
            'template': self.galaxy_map,
            'flux': cy.hyp.BinnedFlux(Defaults.map_E_edge, self.nevts * self.f_astro_north_truth / np.diff(Defaults.map_E_edge)),
            #'flux': cy.hyp.PowerLawFlux(gamma=2.28),
            'sigsub': True,
            'fast_weight': True,
            'dir': cy.utils.ensure_dir('{}/templates/WISE'.format(ana_dir))
        }

        self.conf = conf
        self.trial_runner = cy.get_trial_runner(self.conf)

        ana.save(ana_dir)

        self._prob_reject()

    def _prob_reject(self):
        npix = hp.nside2npix(self.nside)
        acceptance = np.zeros((Defaults.NEbin, npix))
        subana = self.ana[0]
        class dec: dec = np.pi/2 - hp.pix2ang(self.nside, np.arange(npix))[0]
        for i, (emin, emax) in enumerate(zip(Defaults.map_logE_edge, Defaults.map_logE_edge[1:])):
            sig = subana.sig[(subana.sig['log10energy'] > emin) & (subana.sig['log10energy'] < emax)]
            if len(sig) == 0:
                acceptance[i] = np.ones(npix)
            else:
                acc_model = cy.pdf.SinDecAccParameterization(sig)
                acceptance[i] = acc_model(dec, gamma=2.28)
                acceptance[i] /= acceptance[i].max()
        self.prob_reject = acceptance

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

        if np.array(f_diff).size == 1:
            #n_inj = (self.nevts * self.f_astro_north_truth / np.sum(self.galaxy_map * self.prob_reject, axis=1)).sum() * N_yr * f_diff
            n_inj = (self.nevts * self.f_astro_north_truth).sum() * N_yr * f_diff
            self.n_inj = n_inj
            trial, _ = self.trial_runner.get_one_trial(n_inj)

        elif np.array(f_diff).size == self.f_astro_north_truth.size:
            #n_inj = (self.nevts * f_diff / np.sum(self.galaxy_map * self.prob_reject, axis=1)).sum() * N_yr * len(self.ana)
            n_inj = (self.nevts * f_diff).sum() * N_yr
            self.n_inj = n_inj
            trial, _ = self.trial_runner.get_one_trial(n_inj)

        else:
            raise ValueError('Scalar f_diff or f_diff shape equal to f_astro_north_truth')


        # create a sky map of atmospheric nu
        atm_ra = np.hstack([t[0]['ra'] for t in trial])
        atm_dec = np.hstack([t[0]['dec'] for t in trial])
        atm_idx = np.hstack([t[0]['idx'] for t in trial])
        atm_log10energy = np.hstack([t[0]['log10energy'] for t in trial])

        if f_diff > 0:
            astro_ra = np.hstack([t[1]['ra'] for t in trial])
            astro_dec = np.hstack([t[1]['dec'] for t in trial])
            astro_idx = np.hstack([t[1]['idx'] for t in trial])
            astro_log10energy = np.hstack([t[1]['log10energy'] for t in trial])


        astro_maps = np.zeros((Defaults.NEbin, hp.nside2npix(self.nside)))
        atm_maps = np.zeros((Defaults.NEbin, hp.nside2npix(self.nside)))
        for i, (emin, emax) in enumerate(zip(Defaults.map_logE_edge, Defaults.map_logE_edge[1:])):

            if np.any(f_diff > 0):

                atm_mask = (atm_log10energy > emin) & (atm_log10energy < emax) & (atm_dec > (np.pi/2 - Defaults.theta_north)) 
                astro_mask = (astro_log10energy > emin) & (astro_log10energy < emax) & (astro_dec > (np.pi/2 - Defaults.theta_north))
                n_atm = atm_mask.sum() - astro_mask.sum()
                if n_atm < 0:
                    n_atm = 0
                #included_events = np.random.choice(atm_idx[atm_mask], size=n_atm, replace=False)

                # TODO: add in declination selection
                #astro_mask = (astro_log10energy > emin) & (astro_log10energy < emax) & (astro_dec > (np.pi/2 - Defaults.theta_north))
                #included_events = []
                #dec_bands = np.arange(-np.pi/2, np.pi/2 + 0.01, 0.2)
                #for dmin, dmax in zip(dec_bands, dec_bands[1:]):
                #    atm_band_mask = (atm_dec > dmin) & (atm_dec < dmax) & (atm_log10energy > emin) & (atm_log10energy < emax) & (atm_dec > (np.pi/2 - Defaults.theta_north))
                #    astro_band_mask = (astro_dec > dmin) & (astro_dec < dmax) & (astro_log10energy > emin) & (astro_log10energy < emax) & (astro_dec > (np.pi/2 - Defaults.theta_north))
                #    n_atm = atm_band_mask.sum() - astro_band_mask.sum()
                #    if n_atm > 0:
                #        included_events.extend(np.random.choice(atm_idx[atm_band_mask], size=n_atm, replace=False))


                #atm_maps[i] = event2map(atm_ra[included_events], atm_dec[included_events], self.nside)
                #atm_maps[i] = event2map(atm_ra[atm_mask], atm_dec[atm_mask], self.nside)
                atm_maps[i] = event2map(atm_ra[:n_atm], atm_dec[:n_atm], self.nside)

                astro_maps[i] = event2map(astro_ra[astro_mask], astro_dec[astro_mask], self.nside)

            else:
                atm_mask = (atm_log10energy > emin) & (atm_log10energy < emax) & (atm_dec > (np.pi/2 - Defaults.theta_north))
                atm_maps[i] = event2map(atm_ra[atm_mask], atm_dec[atm_mask], self.nside)
                astro_maps[i] = np.zeros(hp.nside2npix(self.nside))

        data_maps = astro_maps + atm_maps
        return data_maps


# TODO: check if this is implemented elsewhere
def event2map(ra, dec, nside=128):
    npix = hp.nside2npix(nside)
    theta = np.pi/2 - dec
    trial_map = np.histogram(hp.ang2pix(nside, theta, ra), bins=np.arange(npix + 1))[0]
    return trial_map
