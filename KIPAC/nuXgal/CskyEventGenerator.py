import os

import csky as cy
from csky.inj import RARandomizer, DecRandomizer
import healpy as hp
import numpy as np
from scipy.stats import gaussian_kde

from . import Defaults
from .DataSpec import data_spec_factory


class NullEnergyPDFRatioEvaluator(cy.pdf.EnergyPDFRatioEvaluator):
    def __call__(self, _mask=None, **params):
        return 1., 1.


class NullEnergyPDFRatioModel(cy.pdf.EnergyPDFRatioModel):
    def __call__(self, ev):
        return NullEnergyPDFRatioEvaluator(ev, self)


class CskyEventGenerator():
    def __init__(self, N_yr, galaxy_sample, gamma=2, Ebinmin=0, Ebinmax=-1, idx_mask=None, mc_background=False):
        """
        Initialize the CskyEventGenerator object.

        Parameters
        ----------
        N_yr : int
            Number of years of data.
        galaxy_sample : GalaxySample
            Instance of the GalaxySample class.
        gamma : float, optional
            Spectral index for the power-law flux. Default is 2.
        Ebinmin : int, optional
            Minimum energy bin index. Default is 0.
        Ebinmax : int, optional
            Maximum energy bin index. Default is -1.
        idx_mask : ndarray, optional
            Mask for the density map. Default is None.
        mc_background : bool, optional
            Flag indicating whether to include Monte Carlo background. Default is False.

        Returns
        -------
        None
        """
        self.galaxyName = galaxy_sample.galaxyName
        self.npix = Defaults.NPIXEL
        self.nside = Defaults.NSIDE
        self.ana_dir = Defaults.NUXGAL_ANA_DIR
        self.gamma = gamma
        self.Ebinmin = Ebinmin
        self.Ebinmax = Ebinmax
        self.log_emin = Defaults.map_logE_edge[Ebinmin]
        self.log_emax = Defaults.map_logE_edge[Ebinmax]
        self.idx_mask = idx_mask

        data_specs = data_spec_factory(Ebinmin, Ebinmax)

        self.dataspec = {
            3: data_specs.ps_3yr,
            10: data_specs.ps_10yr,
            'v4': data_specs.ps_v4,
            'ps_v4': data_specs.ps_v4,
            'estes_10yr': data_specs.estes_10yr,
            'dnn_cascade_10yr': data_specs.dnn_cascade_10yr,
            'nt_v5': data_specs.nt_v5}[N_yr]

        version = {
            'v4': 'version-004-p02',
            'ps_v4': 'version-004-p02',
            'estes_10yr': 'version-001-p03',
            'dnn_cascade_10yr': 'version-001-p01',
            'nt_v5': 'version-005-p01'}[N_yr]

        density_nu = galaxy_sample.density.copy()
        density_nu[idx_mask[0]] = hp.UNSEEN
        density_nu = hp.ma(density_nu)
        self.density_nu = density_nu / density_nu.sum()

        if mc_background:
            for season in self.dataspec:
                season._keep.append('conv')

        # temporary fix to avoid cluster file transfer problem
        uname = os.uname()
        self.ana = cy.get_analysis(cy.selections.repo, version, self.dataspec, analysis_region_template=~self.density_nu.mask)
        self.conf = {
            'ana': self.ana,
            'template': self.density_nu.copy(),
            'flux': cy.hyp.PowerLawFlux(self.gamma),
            'sigsub': True,
        }

        if mc_background:
            inj_conf = {'bg_weight_names': ['conv']}
            self.trial_runner = cy.get_trial_runner(self.conf, inj_conf=inj_conf)
        else:
            self.trial_runner = cy.get_trial_runner(self.conf)

        self._filter_injector_events()

    def _filter_mask_events(self, trial):
        for i, tr in enumerate(trial):
            for j, evts in enumerate(tr):
                pix = hp.ang2pix(Defaults.NSIDE, np.degrees(evts['ra']), np.degrees(evts['dec']), lonlat=True)
                idx = ~np.in1d(pix, self.idx_mask)  # check if events are in masked region
                trial[i][j] = evts[idx]
        return trial

    def _filter_injector_events(self):
        """Remove simulated and background events from csky injectors"""

        for i, injector in enumerate(self.trial_runner.sig_injs):
            sig = injector.sig
            idx = np.ones(len(injector.sig), dtype=bool)

            # energy filter
            idx *= (sig['log10energy'] > self.log_emin) * (sig['log10energy'] < self.log_emax)

            # replace signal events in place
            self.trial_runner.sig_injs[i].flux_weights[~idx] = 0.

        for i, injector in enumerate(self.trial_runner.bg_injs):
            if isinstance(injector, cy.inj.MCBackgroundInjector):
                data = injector.mc
            else:
                data = injector.data
            idx = np.ones(len(data), dtype=bool)

            # no need for spatial filter since the events get RA scrambled
            # we have to remove them in each trial

            # energy filter
            idx *= (data['log10energy'] > self.log_emin) * (data['log10energy'] < self.log_emax)

            # replace signal events in place
            self.trial_runner.bg_injs[i].data = data[idx]

    def updateGamma(self, gamma):
        self.conf['flux'] = cy.hyp.PowerLawFlux(gamma)
        self.conf['fitter_args'] = dict(gamma=gamma)
        self.trial_runner = cy.get_trial_runner(self.conf)

    def SyntheticTrial(self, ninj, keep_total_constant=True, signal_only=False):
        """
        Perform a synthetic trial by generating events and optionally modifying them.

        Parameters
        ----------
        ninj : int
            The number of signal events to inject.
        keep_total_constant : bool, optional
            Whether to keep the total number of events constant by removing injected events.
        signal_only : bool, optional
            Whether to keep only the signal events by removing background events.

        Returns
        -------
        tuple
            A tuple containing the modified events and the number of excess events.
        """
        events, nexc = self.trial_runner.get_one_trial(ninj)

        if keep_total_constant:
            # we want to measure what fraction of events are astrophysical, so remove the number of events injected to hold total events fixed
            events = self._remove_to_keep_constant(events)

        if signal_only:
            for tr in events:
                if len(tr) > 0:
                    tr.pop(0)  # remove non-signal events

        return events, nexc

    def SyntheticTrialMCKDE(self):
        events = []
        if not hasattr(self, 'kdes'):
            self._make_kdes()

        for i, elo in enumerate(range(self.Ebinmin, self.Ebinmax)):
            ehi = elo + 1
            elo = Defaults.map_logE_edge[int(elo)]
            ehi = Defaults.map_logE_edge[int(ehi)]

            sindec = self.kdes[i].resample(self.nevents[i])[0]

            # the kde doesn't hard cut at sindec = 1, so we have to resample until we get valid events
            while np.any(np.abs(sindec) > 1):
                idx = np.abs(sindec) > 1
                sindec[idx] = self.kdes[i].resample(np.sum(idx))[0]
            ra = np.random.uniform(0, 2 * np.pi, len(sindec))
            log10energy = np.random.uniform(elo, ehi, len(sindec))
            events.append([cy.utils.Events(ra=ra, dec=np.arcsin(sindec), log10energy=log10energy)])


        return events

    def _make_kdes(self):
        self.nevents = np.zeros(self.Ebinmax - self.Ebinmin, dtype=int)

        for i, elo in enumerate(range(self.Ebinmin, self.Ebinmax)):
            ehi = elo + 1
            elo = Defaults.map_logE_edge[int(elo)]
            ehi = Defaults.map_logE_edge[int(ehi)]

            for ana in self.ana:
                idx = (ana.data['log10energy'] >= elo) * (ana.data['log10energy'] < ehi)
                self.nevents[i] += len(ana.data['log10energy'][idx])

        self.kdes = []
        for i, elo in enumerate(range(self.Ebinmin, self.Ebinmax)):
            ehi = elo + 1
            elo = Defaults.map_logE_edge[int(elo)]
            ehi = Defaults.map_logE_edge[int(ehi)]

            mc = []
            weight = []
            for bg_inj in self.trial_runner.bg_injs:
                idx = (bg_inj.mc['log10energy'] >= elo) * (bg_inj.mc['log10energy'] < ehi)
                mc.append(bg_inj.mc['sindec'][idx])
                weight.append(bg_inj.probs[0][idx])

            mc = np.concatenate(mc)
            weight = np.concatenate(weight)
            self.kdes.append(gaussian_kde(mc, weights=weight))

    def _remove_to_keep_constant(self, trial):
        """Remove a number of atmospheric events equal to the number of signal events"""
        for tr in trial:
            if len(tr) == 2:
                n_remove = len(tr[1])
            else:
                n_remove = 0
            tr[0] = tr[0][n_remove:]
        return trial

    def SyntheticData(self, ninj):

        events, nexc = self.trial_runner.get_one_trial(ninj)
        countsmap = np.zeros((Defaults.NEbin, self.npix))
        ninj_ebin = np.zeros(Defaults.NEbin)

        for subana, evt in zip(self.ana, events):
            for i in range(Defaults.NEbin):
                elo = Defaults.map_logE_edge[i]
                ehi = Defaults.map_logE_edge[i + 1]
                if len(evt) == 2:
                    evt[1]['energy'] = 10 ** evt[1]['log10energy']  # have to add this column to astro events
                    ninj_ebin[i] += np.sum((evt[1]['log10energy'] > elo) * (evt[1]['log10energy'] < ehi))
                    evt = cy.utils.Arrays.concatenate(evt)
                else:
                    evt = evt[0]
                evt_idx = (evt['log10energy'] > elo) * (evt['log10energy'] < ehi)
                evt_subset = evt[evt_idx]
                ra = np.degrees(evt_subset['ra'])
                dec = np.degrees(evt_subset['dec'])
                pixels = hp.ang2pix(self.nside, ra, dec, lonlat=True)
                #if self.weighted:
                #    weights = pdf_ratio[evt_idx] / (1 + pdf_ratio[evt_idx])
                #else:
                #    weights = np.ones(pixels.size)
                countsmap[i, pixels] += 1
        countsmap[:, Defaults.idx_muon] = 0
        #countsmap[:, self.density_nu==0] = 0
        return countsmap, ninj_ebin
