"""Top level class to generate synthetic events"""

import os
import numpy as np

import healpy as hp
import csky as cy

from . import Defaults
from . import file_utils
from .Generator import AtmGenerator, AstroGenerator_v2, get_dnde_astro
from .Exposure import ICECUBE_EXPOSURE_LIBRARY


class EventGenerator():
    """Class to generate synthetic IceCube events

    This can generate both atmospheric and astrophysical events
    """
    def __init__(self, year='IC86-2012', astroModel=None):
        """C'tor
        """
        self.year = year

        # TODO: set up data set selections
        dsets = {
            'IC79-2010': cy.selections.PSDataSpecs.IC79,
            'IC86-2011': cy.selections.PSDataSpecs.IC86_2011,
            'IC86-2012': cy.selections.PSDataSpecs.IC86v3_2012,
        }

        ana = cy.get_analysis(cy.selections.repo, dsets[year])
        cy.CONF['ana'] = ana
        ana_dir = Defaults.NUXGAL_ANA_DIR

        galaxy_map = hp.read_map(Defaults.GALAXYMAP_FORMAT.format(galaxyName='WISE'))
        galaxy_map[Defaults.idx_muon] = 0.
        galaxy_map /= galaxy_map.sum()
        self.nside = hp.npix2nside(galaxy_map.size)

        self.trial_runner = []
        for emin, emax in zip(Defaults.map_E_edge, Defaults.map_E_edge[1:]):

            conf = {
                'template': galaxy_map,
                'flux': cy.hyp.PowerLawFlux(2.5, energy_range=(emin, emax)),
                'fitter_args': dict(gamma=2.5),
                'sigsub': True,
                'fast_weight': True,
                'dir': cy.utils.ensure_dir('{}/templates/WISE'.format(ana_dir))
            }

            self.trial_runner.append(cy.get_trial_runner(conf))

        # TODO: check if this is the correct number of events
        trial, _ = self.trial_runner[0].get_one_trial(0)
        self.nevts = np.sum([len(t[0]) for t in trial])

        #self.astroModel = None
        #self.Aeff_max = None
        #self._astro_gen = None
        #if astroModel is not None:
        #    self.initializeAstro(astroModel)


    def initializeAstro(self, astroModel):
        """Initialize the event generate for a particular astrophysical model

        Parameters
        ----------
        astroModel : `str`
            The astrophysical model we are using
        """
        self.astroModel = astroModel

        assert (astroModel == 'observed_numu_fraction'), "EventGenerator: incorrect astroModel"

        # Fig 3 of 1908.09551
        self.f_astro_north_truth = np.array([0, 0.00221405, 0.01216614, 0.15222642, 0., 0., 0.]) * 2.
        spectralIndex = 2.28

        aeff = ICECUBE_EXPOSURE_LIBRARY.get_exposure(self.year, spectralIndex)
        self.Aeff_max = aeff.max(1)
        self._astro_gen = AstroGenerator_v2(Defaults.NEbin, aeff=aeff)


    @property
    def atm_gen(self):
        """Astrospheric event generator"""
        return self._atm_gen

    @property
    def astro_gen(self):
        """Astrophysical event generator"""
        return self._astro_gen

    def astroEvent_galaxy(self, intrinsicCounts, normalized_counts_map):
        """Generate astrophysical event maps from a galaxy
        distribution and a number of intrinsice events

        Parameters
        ----------
        density : `np.ndarray`
            Galaxy density map, used as a pdf
        intrinsicCounts : `np.ndarray`
            True number of events, without accounting for Aeff variation

        Returns
        -------
        counts_map : `np.ndarray`
            Maps of simulated events
        """

        self._astro_gen.normalized_counts_map = normalized_counts_map
        self._astro_gen.nevents_expected.set_value(intrinsicCounts, clear_parent=False)
        return self._astro_gen.generate_event_maps(1)[0]



    def atmBG_coszenith(self, eventNumber, energyBin):
        """Generate atmospheric background cos(zenith) distributions

        Parameters
        ----------
        eventNumber : `int`
            Number of events to generate
        energyBin : `int`
            Energy bin to consider

        Returns
        -------
        cos_z : `np.ndarray`
            Array of synthetic cos(zenith) values
        """
        return self._atm_gen.cosz_cdf()[energyBin](np.random.rand(eventNumber))



    def atmEvent(self, duration_year):
        """Generate atmosphere event maps from expected rates per year

        Parameters
        ----------
        duration_year : `float`
            Number of eyars to generate

        Returns
        -------
        counts_map : `np.ndarray`
            Maps of simulated events
        """
        eventnumber_Ebin = np.random.poisson(self._atm_gen.nevents_expected() * duration_year)
        self._atm_gen.nevents_expected.set_value(eventnumber_Ebin, clear_parent=False)
        return self._atm_gen.generate_event_maps(1)[0]



    def SyntheticData(self, N_yr, f_diff):
        """Generate Synthetic Data

        Parameters
        ----------
        N_yr : `float`
            Number of years of data to generate
        f_diff : `float`
            Fraction of astro events w.r.t. diffuse muon neutrino flux,
            f_diff = 1 means injecting astro events that sum up to 100% of diffuse muon neutrino flux
        density_nu : `np.ndarray`
            Background neutrino density map

        Returns
        -----
        counts_map : `np.ndarray`
            Maps of simulated events
        """

        data_maps = []
        for i, tr in enumerate(self.trial_runner):
            trial, _ = tr.get_one_trial(self.nevts * N_yr * f_diff[i], replace_evts=True)

            ra = np.hstack([t[0]['ra'] for t in trial])
            dec = np.hstack([t[0]['dec'] for t in trial])
            atm_map = event2map(ra, dec, self.nside)

            if f_diff[i] > 0:
                ra = np.hstack([t[1]['ra'] for t in trial])
                dec = np.hstack([t[1]['dec'] for t in trial])
                astro_map = event2map(ra, dec, self.nside)

            else:
                astro_map = np.zeros(atm_map.shape)

            data_maps.append(astro_map + atm_map)

        data_maps = np.vstack(data_maps)
        return data_maps


# TODO: check if this is implemented elsewhere
def event2map(ra, dec, nside=128):
    npix = hp.nside2npix(nside)
    theta = np.pi/2 - dec
    trial_map = np.histogram(hp.ang2pix(nside, theta, ra), bins=np.arange(npix + 1))[0]
    return trial_map
