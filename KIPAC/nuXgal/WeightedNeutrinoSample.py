import healpy as hp
import numpy as np

from . import Defaults
from .NeutrinoSample import NeutrinoSample

class WeightedNeutrinoSample(NeutrinoSample):
    def inputTrial(self, trial):
        self.event_list = trial
        self.countsMap()
        self.unweighted_countsmap = self.countsmap.copy()
        self.unweighted_countsmap_fullsky = self.countsmap.copy()
        self.countsmap_fullsky = self.countsmap.copy()

    def countsMap(self):
        countsmap = np.zeros((Defaults.NEbin, Defaults.NPIXEL))
        for i in range(Defaults.NEbin):
            for evt in self.event_list:
                for tr in evt:
                    elo = Defaults.map_logE_edge[i]
                    ehi = Defaults.map_logE_edge[i + 1]
                    idx = (tr['log10energy'] > elo) * (tr['log10energy'] < ehi)
                    ra = np.degrees(tr['ra'][idx])
                    dec = np.degrees(tr['dec'][idx])
                    pixels = hp.ang2pix(Defaults.NSIDE, ra, dec, lonlat=True)
                    countsmap[i, pixels] += 1
        self.countsmap = countsmap


    def updateCountsMap(self, gamma, ana):
        countsmap = np.zeros((Defaults.NEbin, Defaults.NPIXEL))
        for i in range(Defaults.NEbin):
            for evts, subana in zip(self.event_list, ana):
                for tr in evts:
                    pdf_ratio = subana.energy_pdf_ratio_model(tr)(gamma=gamma)[1]
                    elo = Defaults.map_logE_edge[i]
                    ehi = Defaults.map_logE_edge[i + 1]
                    idx = (tr['log10energy'] > elo) * (tr['log10energy'] < ehi)
                    ra = np.degrees(tr['ra'][idx])
                    dec = np.degrees(tr['dec'][idx])
                    pixels = hp.ang2pix(Defaults.NSIDE, ra, dec, lonlat=True)
                    #weights = pdf_ratio[idx] / (1 + pdf_ratio[idx])
                    weights = np.log(pdf_ratio[idx])
                    weights[weights < 0] = 0
                    countsmap[i, pixels] += weights

        self.countsmap = countsmap

    def getEventCounts(self):
        return self.unweighted_countsmap.sum(axis=1)

    def updateMask(self, idx_mask):
        self.idx_mask = idx_mask
        self.f_sky = 1. - len(idx_mask[0]) / float(Defaults.NPIXEL)
        countsmap = self.countsmap_fullsky.copy() + 0. # +0. to convert to float array
        unweighted_countsmap = self.unweighted_countsmap_fullsky.copy() + 0.
        for i in range(Defaults.NEbin):
            countsmap[i][idx_mask] = hp.UNSEEN
            unweighted_countsmap[i][idx_mask] = hp.UNSEEN
        self.countsmap = hp.ma(countsmap)
        self.unweighted_countsmap = hp.ma(countsmap)