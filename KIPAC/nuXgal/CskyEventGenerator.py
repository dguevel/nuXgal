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
        self.ana_dir = Defaults.NUXGAL_ANA_DIR

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

        n_inj = (f_diff * Defaults.f_astro_north_truth * self.nevts).sum() # probably change to the trial_runner.to_ns method

        x = Defaults.map_E_edge
        y = Defaults.f_astro_north_truth * self.nevts * np.sum(self.prob_reject * self.galaxy_map, axis=1) / np.diff(x)

        conf = {
            'template': self.galaxy_map,
            #'flux': cy.hyp.PowerLawFlux(gamma=2.28),
            'flux': cy.hyp.BinnedFlux(x, y),
            'sigsub': True,
            'fast_weight': True,
            'dir': cy.utils.ensure_dir('{}/templates/WISE'.format(self.ana_dir))
        }

        trial_runner = cy.get_trial_runner(conf)
        trial, _ = trial_runner.get_one_trial(n_inj)


        # create a sky map of atmospheric nu
        atm_ra = trial[0][0]['ra']
        atm_dec = trial[0][0]['dec']
        atm_idx = trial[0][0]['idx']
        atm_log10energy = trial[0][0]['log10energy']

        if n_inj > 0:
            astro_ra = trial[0][1]['ra']
            astro_dec = trial[0][1]['dec']
            astro_idx = trial[0][1]['idx']
            astro_log10energy = trial[0][1]['log10energy']

        f_astro = np.zeros(Defaults.f_astro_north_truth.size)

        astro_maps = np.zeros((Defaults.NEbin, hp.nside2npix(self.nside)))
        atm_maps = np.zeros((Defaults.NEbin, hp.nside2npix(self.nside)))
        for i, (emin, emax) in enumerate(zip(Defaults.map_logE_edge, Defaults.map_logE_edge[1:])):

            if n_inj > 0:

                atm_mask = (atm_log10energy > emin) & (atm_log10energy < emax) & (atm_dec > (np.pi/2 - Defaults.theta_north)) 
                astro_mask = (astro_log10energy > emin) & (astro_log10energy < emax) & (astro_dec > (np.pi/2 - Defaults.theta_north))
                f_astro[i] = astro_mask.sum() / (atm_mask.sum() + astro_mask.sum())
                if (atm_mask.sum() + astro_mask.sum()) == 0:
                    f_astro[i] = 1.
                n_atm = int(atm_mask.sum() * (1 - f_astro[i]))
                if self.nevts[i] == 0:
                    f_astro[i] = 1
                else:
                    f_astro[i] = astro_mask.sum() / self.nevts[i]
                n_atm = int(atm_mask.sum() * (1 - f_astro[i]))

                if n_atm < 0:
                    n_atm = 0
                included_events = np.random.choice(atm_idx[atm_mask], size=n_atm, replace=False)

                # TODO: add in declination selection
                #astro_mask = (astro_log10energy > emin) & (astro_log10energy < emax) & (astro_dec > (np.pi/2 - Defaults.theta_north))
                #included_events = []
                #dec_bands = np.arange(-np.pi/2, np.pi/2 + 0.01, 0.2)
                #for dmin, dmax in zip(dec_bands, dec_bands[1:]):
                #    atm_band_mask = (atm_dec > dmin) & (atm_dec < dmax) & (atm_log10energy > emin) & (atm_log10energy < emax) & (atm_dec > (np.pi/2 - Defaults.theta_north))
                #    astro_band_mask = (astro_dec > dmin) & (astro_dec < dmax) & (astro_log10energy > emin) & (astro_log10energy < emax) & (astro_dec > (np.pi/2 - Defaults.theta_north))
                #    n_atm = atm_band_mask.sum() - astro_band_mask.sum()
                #    #n_atm = int(atm_band_mask.sum() * (1 - f_astro[i]))
                #    if n_atm > 0:
                #        included_events.extend(np.random.choice(atm_idx[atm_band_mask], size=n_atm, replace=False))


                atm_maps[i] = event2map(atm_ra[included_events], atm_dec[included_events], self.nside)
                #atm_maps[i] = event2map(atm_ra[atm_mask], atm_dec[atm_mask], self.nside)
                #atm_maps[i] = event2map(atm_ra[atm_mask][:n_atm], atm_dec[atm_mask][:n_atm], self.nside)
                #atm_maps[i] = event2map(atm_ra[atm_mask][:n_atm[i]], atm_dec[atm_mask][:n_atm[i]], self.nside)

                astro_maps[i] = event2map(astro_ra[astro_mask], astro_dec[astro_mask], self.nside)

            else:
                atm_mask = (atm_log10energy > emin) & (atm_log10energy < emax) & (atm_dec > (np.pi/2 - Defaults.theta_north))
                atm_maps[i] = event2map(atm_ra[atm_mask], atm_dec[atm_mask], self.nside)
                astro_maps[i] = np.zeros(hp.nside2npix(self.nside))

        print(f_astro[1:4])
        data_maps = astro_maps + atm_maps
        return data_maps


# TODO: check if this is implemented elsewhere
def event2map(ra, dec, nside=128):
    npix = hp.nside2npix(nside)
    theta = np.pi/2 - dec
    trial_map = np.histogram(hp.ang2pix(nside, theta, ra), bins=np.arange(npix + 1))[0]
    return trial_map
