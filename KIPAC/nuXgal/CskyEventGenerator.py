import os

import csky as cy
import healpy as hp
import numpy as np

from . import Defaults

class CskyEventGenerator():
    def __init__(self, N_yr, density_nu, gamma=2, weighted=True):
        self.npix = density_nu.size
        self.nside = hp.npix2nside(self.npix)
        self.ana_dir = Defaults.NUXGAL_ANA_DIR
        self.gamma = gamma
        self.weighted = weighted

        self.dataspec = {
            3: cy.selections.PSDataSpecs.ps_3yr,
            10: cy.selections.PSDataSpecs.ps_10yr}[N_yr][0]

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
            'dir': cy.utils.ensure_dir(os.path.join('{}', 'templates', 'WISE').format(self.ana_dir))
        }
        self.trial_runner = cy.get_trial_runner(self.conf)


    def SyntheticData(self, ninj):

        events, nexc = self.trial_runner.get_one_trial(ninj)
        countsmap = np.zeros((Defaults.NEbin, self.npix))

        for subana, evt in zip(self.ana, events):
            for i in range(Defaults.NEbin):
                elo = Defaults.map_logE_edge[i]
                ehi = Defaults.map_logE_edge[i + 1]
                if len(evt) == 2:
                    evt[1]['energy'] = 10 ** evt[1]['log10energy']# have to add this column to astro events
                    evt = cy.utils.Arrays.concatenate(evt)
                else:
                    evt = evt[0]
                pdf_ratio = subana.energy_pdf_ratio_model(evt)(gamma=self.gamma)[0]
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

        return countsmap
