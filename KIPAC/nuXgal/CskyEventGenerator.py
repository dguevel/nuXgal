import os

import csky as cy
from csky.inj import RARandomizer, DecRandomizer
import healpy as hp
import numpy as np

from . import Defaults
from .DataSpec import data_spec_factory


class NullEnergyPDFRatioEvaluator(cy.pdf.EnergyPDFRatioEvaluator):
    def __call__(self, _mask=None, **params):
        return 1., 1.


class NullEnergyPDFRatioModel(cy.pdf.EnergyPDFRatioModel):
    def __call__(self, ev):
        return NullEnergyPDFRatioEvaluator(ev, self)


class CskyEventGenerator():
    def __init__(self, N_yr, galaxy_sample, gamma=2, Ebinmin=0, Ebinmax=-1, idx_mask=None):
        self.galaxyName = galaxy_sample.galaxyName
        self.npix = Defaults.NPIXEL
        self.nside = Defaults.NSIDE
        self.ana_dir = Defaults.NUXGAL_ANA_DIR
        self.gamma = gamma
        self.log_emin = Defaults.map_logE_edge[Ebinmin]
        self.log_emax = Defaults.map_logE_edge[Ebinmax]
        self.idx_mask = idx_mask

        data_specs = data_spec_factory(Ebinmin, Ebinmax)
        self.dataspec = {
            3: data_specs.ps_3yr,
            10: data_specs.ps_10yr,
            'v4': data_specs.ps_v4,
            'estes_10': data_specs.estes_10yr}[N_yr]

        density_nu = galaxy_sample.density.copy()
        density_nu[idx_mask[0]] = hp.UNSEEN
        density_nu = hp.ma(density_nu)
        self.density_nu = density_nu / density_nu.sum()

        # temporary fix to avoid cluster file transfer problem
        uname = os.uname()
        if ('cobalt' in uname.nodename) or ('tyrell' in uname.nodename):
            self.ana = cy.get_analysis(cy.selections.repo, Defaults.ANALYSIS_VERSION, self.dataspec, dir=self.ana_dir, analysis_region_template=~self.density_nu.mask)
            #self.ana = cy.get_analysis(cy.selections.repo, Defaults.ANALYSIS_VERSION, self.dataspec, analysis_region_template=~self.density_nu.mask)
            self.ana.save(self.ana_dir)

        else:
            self.ana = cy.get_analysis(cy.selections.repo, Defaults.ANALYSIS_VERSION, self.dataspec, analysis_region_template=~self.density_nu.mask)

        self.conf = {
            'ana': self.ana,
            'template': self.density_nu.copy(),
            'flux': cy.hyp.PowerLawFlux(self.gamma),
            'sigsub': True,
            'fast_weight': True,
            'randomizers': [
                RARandomizer(),
                DecRandomizer()
            ]
        }
        if ('cobalt' in uname.nodename) or ('tyrell' in uname.nodename):
            self.conf['dir'] = cy.utils.ensure_dir(os.path.join('{}', 'templates', self.galaxyName).format(self.ana_dir))
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
            data = injector.data
            idx = np.ones(len(injector.data), dtype=bool)

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
        events, nexc = self.trial_runner.get_one_trial(ninj)

        if keep_total_constant:
            # we want to measure what fraction of events are astrophysical, so remove the number of events injected to hold total events fixed
            events = self._remove_to_keep_constant(events)

        if signal_only:
            for tr in events:
                if len(tr) > 0:
                    tr.pop(0)  # remove non-signal events

        return events, nexc

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
