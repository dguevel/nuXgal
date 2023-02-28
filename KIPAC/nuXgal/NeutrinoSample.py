"""Classes and functions to manage neutrino event samples"""

import numpy as np

import healpy as hp

from . import Defaults

from . import file_utils

from .Exposure import ICECUBE_EXPOSURE_LIBRARY, Aeff

from .plot_utils import FigureDict


class NeutrinoSample():
    """Neutrino event sample class"""

    def __init__(self):
        """C'tor"""
        self.countsmap = None
        self.idx_mask = None
        self.f_sky = 1.
        self.countsmap_fullsky = None


    def inputTrial(self, trial, nyear):
        self.exposure = Aeff(nyear)
        self.event_list = trial
        self.countsMap()
        self.countsmap_fullsky = self.countsmap.copy()
        #self.fluxMap()
        #self.fluxmap_fullsky = self.fluxmap.copy()
        self.fluxmap = np.zeros(self.countsmap.shape)
        self.fluxmap_fullsky = self.fluxmap.copy()

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
                    bins = np.arange(Defaults.NPIXEL+1)
                    countsmap[i] += np.histogram(pixels, bins=bins)[0]
        self.countsmap = countsmap

    def fluxMap(self):
        #w_cross = [np.zeros(Defaults.MAX_L + 1) for i in range(Defaults.NEbin)]
        dOmega = hp.nside2pixarea(Defaults.NSIDE)
        fluxmap = np.zeros((Defaults.NEbin, Defaults.NPIXEL))
        for i in range(Defaults.NEbin):
            for evt in self.event_list:
                for j, (elo_micro, ehi_micro) in enumerate(zip(Defaults.logE_microbin_edge, Defaults.logE_microbin_edge[1:])):
                    # build exposure map for this micro energy bin
                    elo, ehi = Defaults.map_logE_edge[i], Defaults.map_logE_edge[i+1]
                    exp_pixels = np.arange(Defaults.NPIXEL)
                    exp_dec = np.pi/2 - hp.pix2ang(Defaults.NSIDE, exp_pixels)[0]
                    eMid = (elo_micro+ehi_micro)/2
                    exposuremap = self.exposure(eMid, np.sin(exp_dec))
                    exposuremap[Defaults.idx_muon] = hp.UNSEEN
                    exposuremap = hp.ma(exposuremap)

                    # build counts map for this micro energy bin
                    countsmap = np.zeros(Defaults.NPIXEL)
                    for tr in evt:
                        idx = (tr['log10energy'] > elo_micro) * (tr['log10energy'] < ehi_micro) * (tr['log10energy'] > elo) * (tr['log10energy'] < ehi)
                        if np.sum(idx) > 0:
                            ra = np.degrees(tr['ra'][idx])
                            dec = np.degrees(tr['dec'][idx])
                            pixels = hp.ang2pix(Defaults.NSIDE, ra, dec, lonlat=True)
                            bins = np.arange(Defaults.NPIXEL+1)
                            countsmap += np.histogram(pixels, bins=bins)[0]
                    #countsmap[Defaults.idx_muon] = hp.UNSEEN
                    #countsmap = hp.ma(countsmap)
                    fluxmap[i] += countsmap / (dOmega * exposuremap) # no need for dE if we're summing over microE


        self.fluxmap = fluxmap

    def inputCountsmap(self, countsmap):
        """Set the counts map

        Parameters
        ----------
        countsmap : `np.ndarray`
            The input map
        """
        self.countsmap = countsmap
        self.countsmap_fullsky = countsmap


    def inputData(self, countsmappath):
        """Set the counts map for a filename

        Parameters
        ----------
        countsmappath : `str`
            Path to the counts map
        """
        self.countsmap = file_utils.read_maps_from_fits(countsmappath, Defaults.NEbin)
        self.countsmap_fullsky = self.countsmap

    def updateMask(self, idx_mask):
        """Set the mask used to analyze the data

        Parameters
        ----------
        idx_mask : `np.ndarray`
            Masks
        """
        self.idx_mask = idx_mask
        self.f_sky = 1. - len(idx_mask[0]) / float(Defaults.NPIXEL)
        countsmap = self.countsmap_fullsky.copy() + 0. # +0. to convert to float array
        fluxmap = self.fluxmap_fullsky.copy() + 0.
        for i in range(Defaults.NEbin):
            countsmap[i][idx_mask] = hp.UNSEEN
            fluxmap[i][idx_mask] = hp.UNSEEN
        self.countsmap = hp.ma(countsmap)
        self.fluxmap = hp.ma(fluxmap)

    def getEventCounts(self):
        """Return the number of counts in each energy bin"""
        return self.countsmap.sum(axis=1)


    def getIntensity(self, dt_years, spectralIndex=3.7, year='IC86-2012'):
        """Compute the intensity / energy flux of the neutirno sample"""
        exposuremap = ICECUBE_EXPOSURE_LIBRARY.get_exposure(year, spectralIndex)
        fluxmap = np.divide(self.countsmap, exposuremap,
                            out=np.zeros_like(self.countsmap),
                            where=exposuremap != 0)
        intensity = fluxmap.sum(axis=1) / (10.**Defaults.map_logE_center * np.log(10.) *\
                                               Defaults.map_dlogE) / (dt_years * Defaults.DT_SECONDS) /\
                                               (4 * np.pi * self.f_sky)  / 1e4 ## exposure map in m^2
        return intensity

    def getOverdensity(self):
        """Compute and return the overdensity maps"""
        overdensity = [self.countsmap[i] / self.countsmap[i].mean() - 1. for i in range(Defaults.NEbin)]
        return overdensity

    def getFluxOverdensity(self):
        """Compute and return the overdensity maps"""
        overdensity = [self.fluxmap[i] / self.fluxmap[i].mean() - 1. for i in range(Defaults.NEbin)]
        return overdensity

    def getAlm(self):
        """Compute and return the alms"""
        overdensity = self.getOverdensity()
        alm = [hp.sphtfunc.map2alm(overdensity[i]) for i in range(Defaults.NEbin)]
        return alm

    def getPowerSpectrum(self):
        """Compute and return the power spectrum of the neutirno sample"""
        overdensity = self.getOverdensity()
        w_auto = [hp.sphtfunc.anafast(overdensity[i]) / self.f_sky for i in range(Defaults.NEbin)]
        return w_auto


    def getCrossCorrelationMaps(self, overdensityMap_g):
        """Compute the cross correlation between the overdensity map and a counts map"""
        overdensity = self.getOverdensity()
        w_cross = [hp.sphtfunc.anafast(overdensity[i], overdensityMap_g[i]) / self.f_sky for i in range(Defaults.NEbin)]
        return w_cross

    def getCrossCorrelation(self, alm_g):
        """Compute and return cross correlation between the overdensity map and a counts map

        Parameters
        ----------
        alm_g : `np.ndarray`
            The alm for the sample are correlating against

        Returns
        -------
        w_cross : `np.ndarray`
            The cross correlation
        """
        overdensity = self.getOverdensity()
        alm_nu = [hp.sphtfunc.map2alm(overdensity[i]) for i in range(Defaults.NEbin)]
        w_cross = [hp.sphtfunc.alm2cl(alm_nu[i], alm_g) / self.f_sky for i in range(Defaults.NEbin)]
        return np.array(w_cross)

    def getFluxCrossCorrelation(self, alm_g):
        """Compute and return cross correlation between the overdensity map and a counts map

        Parameters
        ----------
        alm_g : `np.ndarray`
            The alm for the sample are correlating against

        Returns
        -------
        w_cross : `np.ndarray`
            The cross correlation
        """

        overdensity = self.getFluxOverdensity()
        alm_nu = [hp.sphtfunc.map2alm(overdensity[i]) for i in range(Defaults.NEbin)]
        w_cross = [hp.sphtfunc.alm2cl(alm_nu[i], alm_g) / self.f_sky for i in range(Defaults.NEbin)]
        return np.array(w_cross)

    def plotCountsmap(self, testfigpath):
        """Plot and save the maps"""
        figs = FigureDict()
        figs.mollview_maps('countsmap', self.countsmap)
        figs.save_all(testfigpath, 'pdf')

    def updateCountsMap(self, *args, **kwargs):
        pass

    def updateFluxMap(self, *args, **kwargs):
        pass

    #def getCrossCorrelation_countsmap(self, countsmap, overdensityMap_g, idx_mask):
    #    """Compute the cross correlation between the overdensity map and an atm counts map"""
    #    w_cross = np.zeros((Defaults.NEbin, Defaults.NCL))
    #    for i in range(Defaults.NEbin):
    #        overdensitymap_nu = Utilityfunc.overdensityMap_mask(countsmap[i], idx_mask)
    #        overdensitymap_nu[idx_mask] = hp.UNSEEN
    #        w_cross[i] = hp.sphtfunc.anafast(overdensitymap_nu, overdensityMap_g)
    #    return w_cross
