import os

import csky as cy
import healpy as hp
import numpy as np

from . import Defaults
from .DataSpec import ps_3yr, ps_10yr

class CskyEventGenerator():
    def __init__(self, N_yr, density_nu, galaxyName, gamma=2, weighted=True, Ebinmin=0, Ebinmax=-1):
        self.galaxyName = galaxyName
        self.npix = density_nu.size
        self.nside = hp.npix2nside(self.npix)
        self.ana_dir = Defaults.NUXGAL_ANA_DIR
        self.gamma = gamma
        self.weighted = weighted
        self.emin = Defaults.map_logE_edge[Ebinmin]
        self.emax = Defaults.map_logE_edge[Ebinmax]

        self.dataspec = {
            3: ps_3yr,
            10: ps_10yr}[N_yr][0]

        density_nu = density_nu.copy()
        density_nu[Defaults.idx_muon] = 0
        self.density_nu = density_nu / density_nu.sum()

        self.ana = cy.get_analysis(cy.selections.repo, 'version-003-p03', self.dataspec)
        self.conf = {
            'ana': self.ana,
            'template': density_nu.copy(),
            'flux': cy.hyp.PowerLawFlux(self.gamma),
            'sigsub': True,
            'fast_weight': True,
            'dir': cy.utils.ensure_dir(os.path.join('{}', 'templates', self.galaxyName).format(self.ana_dir))
        }
        self.trial_runner = cy.get_trial_runner(self.conf)
        self.getBlurredTemplate(load=False)

    def updateGamma(self, gamma):
        self.conf['flux'] = cy.hyp.PowerLawFlux(gamma)
        self.trial_runner = cy.get_trial_runner(self.conf)
        self.getBlurredTemplate(load=False)

    def getBlurredTemplate(self, save=True, load=True):
        """Save the acceptance weighted PSF smeared template."""

        if load:
            fname = Defaults.BLURRED_GALAXYMAP_FORMAT.format(galaxyName=self.galaxyName)
            return np.load(fname)
        sigma_bins = np.arange(0.2, 3.1, .01)
        smoothed_template = np.zeros((Defaults.NEbin, self.npix))
        for j in np.arange(Defaults.NEbin):
            for i, sig_inj in enumerate(self.trial_runner.sig_injs):
                elo = Defaults.map_logE_edge[j]
                ehi = Defaults.map_logE_edge[j+1]
                idx = (sig_inj.sig['log10energy'] > elo) * (sig_inj.sig['log10energy'] < ehi)
                energy_weights = sig_inj.flux_weights[idx]
                sigma_err = np.degrees(sig_inj.sig['sigma'][idx])
                weights, _ = np.histogram(sigma_err, bins=sigma_bins, normed=True, weights=energy_weights)
                smoothed = np.zeros((weights.size, self.density_nu.size))
                for k, sigma in enumerate(sigma_bins[:-1]):
                    smoothed[k] = hp.smoothing(self.density_nu, sigma=np.radians(sigma)) * weights[k]
                smoothed_template[j] += self.trial_runner.sig_inj_probs[i] * np.sum(smoothed, axis=0)
            smoothed_template[j] /= smoothed_template[j].sum()
        if save:
            fname = Defaults.BLURRED_GALAXYMAP_FORMAT.format(galaxyName=self.galaxyName)
            np.save(fname, smoothed_template)
        return smoothed_template


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
                pdf_ratio = subana.energy_pdf_ratio_model(evt)(gamma=self.gamma)[1]
                evt_idx = (evt['log10energy'] > elo) * (evt['log10energy'] < ehi)
                evt_subset = evt[evt_idx]
                ra = np.degrees(evt_subset['ra'])
                dec = np.degrees(evt_subset['dec'])
                pixels = hp.ang2pix(self.nside, ra, dec, lonlat=True)
                if self.weighted:
                    weights = pdf_ratio[evt_idx] / (1 + pdf_ratio[evt_idx])
                else:
                    weights = np.ones(pixels.size)
                countsmap[i, pixels] += weights
        countsmap[:, Defaults.idx_muon] = 0
        #countsmap[:, self.density_nu==0] = 0
        return countsmap, ninj_ebin
