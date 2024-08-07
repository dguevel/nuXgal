"""Classes and functions to manage neutrino event samples"""

import numpy as np
import scipy
import healpy as hp
import os
import tempfile
try:
    from ispice import ispice
except ImportError:
    print("PolSpice not installed, experimental PolSpice cross correlation will raise error")
from astropy.io import fits

from . import Defaults

from . import file_utils

from .Exposure import ICECUBE_EXPOSURE_LIBRARY

from .plot_utils import FigureDict


class NeutrinoSample():
    """Neutrino event sample class"""

    def __init__(self, nyr='ps_v4'):
        """C'tor"""
        self.countsmap = None
        self.idx_mask = None
        self.f_sky = 1.
        self.countsmap_fullsky = None
        self._effective_area = None

        self.bl = np.zeros((Defaults.NEbin, Defaults.NCL))
        self.bl_fnames = []
        for ebin in range(Defaults.NEbin):
            bl_fname = Defaults.BEAM_FNAME_FORMAT.format(year=nyr, ebin=ebin)
            self.bl_fnames.append(bl_fname)
            self.bl[ebin] = np.load(bl_fname)

    def inputTrial(self, trial, ana):
        self.event_list = trial
        self.build_aeff_matrix(ana)
        self.countsMap()
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
                    loge = tr['log10energy'][idx]
                    #area = self.lookup_aeff(np.sin(np.radians(dec)), loge)
                    #energy = 10**loge
                    pixels = hp.ang2pix(Defaults.NSIDE, ra, dec, lonlat=True)
                    bins = np.arange(Defaults.NPIXEL+1)
                    #countsmap[i] += np.histogram(pixels, bins=bins, weights=energy/area)[0]
                    countsmap[i] += np.histogram(pixels, bins=bins)[0]
        self.countsmap = countsmap

    def build_aeff_matrix(self, ana):
        #sindec_bins = ana[0].spec._bins_sindec
        sindec_bins = np.linspace(-1, 1, 41)
        dsindec = np.diff(sindec_bins)
        loge_bins = np.linspace(2.5, 5.5, 4)
        dlogE = 1
        #dlogE = np.diff(ana[0].spec._bins_logenergy)[0]
        #loge_bins = ana[0].spec._bins_logenergy
        ra, dec = hp.pix2ang(Defaults.NSIDE, np.arange(Defaults.NPIXEL), lonlat=True)
        aeff = np.zeros((len(sindec_bins)-1, len(loge_bins)-1))

        for subana in ana:
            
            solid_angle = 2*np.pi*(dsindec)
            sr_idx = np.digitize(np.sin(subana.sig.dec), sindec_bins)
            sr_idx = np.clip(sr_idx, 0, len(dsindec)-1)
            solid_angle = 2 * np.pi * dsindec[sr_idx]
            
            area = 1 / (1e4*np.log(10)) * (subana.sig.oneweight / (subana.sig.true_energy * solid_angle * dlogE))
            aeff += np.histogram2d(np.sin(subana.sig.dec), np.log10(subana.sig.true_energy), bins=(sindec_bins, loge_bins), weights=area)[0]

        self.sindec_bins = sindec_bins
        self.loge_bins = loge_bins
        self.aeff_matrix = aeff

    def lookup_aeff(self, sindec, loge):
        sindec_idx = np.clip(np.digitize(sindec, self.sindec_bins), 0, len(self.sindec_bins)-2)
        loge_idex = np.clip(np.digitize(loge, self.loge_bins), 0, len(self.loge_bins)-2)
        return self.aeff_matrix[sindec_idx, loge_idex]
    
    def build_aeff_map(self):
        ra, dec = hp.pix2ang(Defaults.NSIDE, np.arange(Defaults.NPIXEL), lonlat=True)
        self.effective_area_map = np.zeros((self.loge_bins.size - 1, Defaults.NPIXEL))
        for i in range(len(self.loge_bins) - 1):
            sindec_idx = np.clip(np.digitize(np.sin(np.radians(dec)), self.sindec_bins), 0, len(self.sindec_bins)-2)
            self.effective_area_map[i] = self.aeff_matrix[sindec_idx, i]

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
        countsmap = self.countsmap_fullsky.copy().astype(float)
        for i in range(Defaults.NEbin):
            countsmap[i][idx_mask] = hp.UNSEEN
        self.countsmap = hp.ma(countsmap)

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

    def getCrossCorrelation(self, galaxy_sample, acceptance=None):
        """Compute and return cross correlation between the overdensity map and a counts map

        Parameters
        ----------
        galaxy_sample : `KIPAC.nuXgal.GalaxySample.GalaxySample`
            The alm for the sample are correlating against

        Returns
        -------
        w_cross : `np.ndarray`
            The cross correlation
        """

        if acceptance is None:
            acceptance = np.ones_like(self.countsmap)
        countsmap = self.countsmap / acceptance
        overdensity_nu = [countsmap[i] / countsmap[i].mean() - 1. for i in range(Defaults.NEbin)]
        w_cross = np.zeros((Defaults.NEbin, Defaults.NCL))
        for i in range(Defaults.NEbin):
            overdensityalm_nu = hp.map2alm(overdensity_nu[i])
            w_cross[i] = hp.alm2cl(overdensityalm_nu, galaxy_sample.overdensityalm, lmax=Defaults.MAX_L) / self.f_sky
        return w_cross

    def getCrossCorrelationEbin(self, galaxy_sample, ebin, acceptance=None):
        """Compute and return cross correlation between the overdensity map and a counts map for one energy bin

        Parameters
        ----------
        alm_g : `np.ndarray`
            The alm for the sample are correlating against

        Returns
        -------
        w_cross : `np.ndarray`
            The cross correlation
        """

        if acceptance is None:
            acceptance = np.ones_like(self.countsmap)
        countsmap = self.countsmap / acceptance
        overdensity_nu = [countsmap[i] / countsmap[i].mean() - 1. for i in range(Defaults.NEbin)]
        overdensityalm_nu = hp.map2alm(overdensity_nu[ebin])
        overdensityalm_gal = galaxy_sample.overdensityalm
        w_cross = hp.sphtfunc.alm2cl(overdensityalm_nu, overdensityalm_gal, lmax=Defaults.MAX_L) / self.f_sky# / self.bl[ebin] / pix_window ** 2
        return w_cross

    def getCrossCorrelationPolSpice(self, galaxy_sample, ana):
        """Compute and return cross correlation between the overdensity map 
        and a galaxy map using PolSpice

        Parameters
        ----------
        galaxy_sample : `KIPAC.nuXgal.GalaxySample.GalaxySample`
            The galaxy sample for the cross correlation

        Returns
        -------
        w_cross : `np.ndarray`
            The cross correlation
        """

        # initialize the cross correlation array
        w_cross = np.zeros((Defaults.NEbin, Defaults.NCL))

        # get the neutrino overdensity
        nu_overdensity = self.getOverdensity()

        # create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
        #if True:
        #    temp_dir = '/home/dguevel/temp'

            # write the galaxy map and mask to disk
            galaxy_map_fname = os.path.join(
                temp_dir,
                'galaxy_map.fits'
            )
            galaxy_mask_fname = os.path.join(
                temp_dir,
                'galaxy_mask.fits'
            )

            hp.write_map(galaxy_map_fname, galaxy_sample.overdensity)
            galaxy_mask = np.zeros_like(galaxy_sample.overdensity, dtype=bool)
            galaxy_mask[galaxy_sample.mask()] = True
            galaxy_mask = ~galaxy_mask
            hp.write_map(galaxy_mask_fname, galaxy_mask, dtype=int)

            # loop over energy bins
            for ebin in range(Defaults.NEbin):

                # write the neutrino map and mask to disk
                neutrino_map_fname = os.path.join(
                    temp_dir,
                    'neutrino_map_ebin{}.fits'.format(ebin)
                )
                neutrino_mask_fname = os.path.join(
                    temp_dir,
                    'neutrino_mask_ebin{}.fits'.format(ebin)
                )
                neutrino_weight_fname = os.path.join(
                    temp_dir,
                    'neutrino_weight_ebin{}.fits'.format(ebin)
                )
                cl_out_fname = os.path.join(
                    temp_dir,
                    'cl_ebin{}.fits'.format(ebin)
                )

                hp.write_map(neutrino_map_fname, nu_overdensity[ebin])
                neutrino_mask = np.zeros_like(self.countsmap[ebin], dtype=bool)
                neutrino_mask[self.idx_mask] = True
                neutrino_mask = ~neutrino_mask
                hp.write_map(neutrino_mask_fname, neutrino_mask, dtype=int)
                #weight = 1 / np.maximum(self.effective_area(ana)[ebin], 0)
                #weight[np.isnan(weight)] = 0
                #weight = np.maximum(self.effective_area(ana)[ebin], 0)
                #hp.write_map(neutrino_weight_fname, weight)

                # get beam file name
                beam_fname = self.bl_fnames[ebin]

                # run PolSpice
                ispice(
                    mapin1=neutrino_map_fname,
                    mapfile2=galaxy_map_fname,
                    maskfile1=neutrino_mask_fname,
                    maskfile2=galaxy_mask_fname,
                    #weightfile1=neutrino_weight_fname,
                    beam_file1=beam_fname,
                    clout=cl_out_fname,
                    apodizesigma=180,
                    thetamax=180,
                    subav=True,
                    subdipole=True,
                )

                # read the output and load into the cross correlation array
                w_cross[ebin] = hp.read_cl(cl_out_fname)

        return w_cross


    def getCrossCorrelationPolSpiceEbin(self, galaxy_sample, ebin, ana):
        """Compute and return cross correlation between the overdensity map 
        and a galaxy map using PolSpice

        Parameters
        ----------
        galaxy_sample : `KIPAC.nuXgal.GalaxySample.GalaxySample`
            The galaxy sample for the cross correlation

        Returns
        -------
        w_cross : `np.ndarray`
            The cross correlation
        """

        # get the neutrino overdensity
        nu_overdensity = self.getOverdensity()

        # create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:

            # write the galaxy map and mask to disk
            galaxy_map_fname = os.path.join(
                temp_dir,
                'galaxy_map.fits'
            )
            galaxy_mask_fname = os.path.join(
                temp_dir,
                'galaxy_mask.fits'
            )

            hp.write_map(galaxy_map_fname, galaxy_sample.overdensity)
            galaxy_mask = np.zeros_like(galaxy_sample.overdensity, dtype=bool)
            galaxy_mask[galaxy_sample.mask()] = True
            galaxy_mask = ~galaxy_mask
            hp.write_map(galaxy_mask_fname, galaxy_mask, dtype=int)


            # write the neutrino map and mask to disk
            neutrino_map_fname = os.path.join(
                temp_dir,
                'neutrino_map_ebin{}.fits'.format(ebin)
            )
            neutrino_mask_fname = os.path.join(
                temp_dir,
                'neutrino_mask_ebin{}.fits'.format(ebin)
            )
            neutrino_weight_fname = os.path.join(
                temp_dir,
                'neutrino_weight_ebin{}.fits'.format(ebin)
            )
            neutrino_cov_fname = os.path.join(
                temp_dir,
                'neutrino_cov_ebin{}.fits'.format(ebin)
            )

            cl_out_fname = os.path.join(
                temp_dir,
                'cl_ebin{}.fits'.format(ebin)
            )

            hp.write_map(neutrino_map_fname, nu_overdensity[ebin])
            neutrino_mask = np.zeros_like(self.countsmap[ebin], dtype=bool)
            neutrino_mask[self.idx_mask] = True
            neutrino_mask = ~neutrino_mask
            hp.write_map(neutrino_mask_fname, neutrino_mask, dtype=int)
            #weight = 1 / np.maximum(self.effective_area(ana)[ebin], 0)
            #weight[np.isnan(weight)] = 0
            #weight = np.maximum(self.effective_area(ana)[ebin], 0)
            #hp.write_map(neutrino_weight_fname, weight)

            # get beam file name
            beam_fname = self.bl_fnames[ebin]

            # run PolSpice
            ispice(
                mapin1=neutrino_map_fname,
                mapfile2=galaxy_map_fname,
                maskfile1=neutrino_mask_fname,
                maskfile2=galaxy_mask_fname,
                #weightfile1=neutrino_weight_fname,
                beam_file1=beam_fname,
                clout=cl_out_fname,
                apodizesigma=180,
                thetamax=180,
                subav=True,
                subdipole=True,
                covfileout=neutrino_cov_fname,
            )

            # read the output and load into the cross correlation array
            w_cross = hp.read_cl(cl_out_fname)
            with fits.open(neutrino_cov_fname) as hdul:
                w_cov = hdul[0].data[0]

        return w_cross, w_cov

    def plotCountsmap(self, testfigpath):
        """Plot and save the maps"""
        figs = FigureDict()
        figs.mollview_maps('countsmap', self.countsmap)
        figs.save_all(testfigpath, 'pdf')

    def effective_area(self, ana):
        if self._effective_area is None:
            self._effective_area = self.calc_effective_area(ana)
            for i in range(Defaults.NEbin):
                self._effective_area[i] = hp.smoothing(self._effective_area[i], fwhm=5*np.pi/180)
        return self._effective_area

    def calc_effective_area(self, ana):

        '''
        Note the following:

        *a=ana.anas[-1] may not contain the entire dataset. For tracks, it will only use the last season whereas DNNCascade
        loads as one season so the entire dataset is used.

        *dlogE is the logarithmic energy bin width

        *solid_angle represents the portion of the sky that we are observing with the detector

        *In the area eqn defined below, 1/ (1e4*np.log(10)) comes from eqn 5.12 in the thesis linked at the start
        of this notebook. The 1e4 accounts for converting from cm^2 -> m^2 and np.log(10) accounts for the ln(10) when
        differentiating dlog(E)
        '''

        dsindec = 0.05
        sindec_bins = np.arange(-1, 1.1, dsindec)
        effective_area_map = np.zeros((Defaults.NEbin, Defaults.NPIXEL))
        ra, dec = hp.pix2ang(Defaults.NSIDE, np.arange(Defaults.NPIXEL), lonlat=True)
        gammas = np.arange(2, 4, 0.1)

        for i, (elo, ehi) in enumerate(zip(Defaults.map_logE_edge[:-1], Defaults.map_logE_edge[1:])):
            for subana in ana:
                mask = subana.sig.log10energy >= elo
                mask &= subana.sig.log10energy < ehi

                dlogE = ehi - elo
                dE = 10**ehi - 10**elo
                solid_angle = 2*np.pi*(dsindec)
                for gamma in gammas:
                    area = subana.sig.oneweight[mask] * subana.sig.true_energy[mask]**-gamma / solid_angle / dE
                    hist, bins = np.histogram(np.sin(subana.sig.dec[mask]), bins=sindec_bins, weights=area)
                    bins_center = (bins[1:] + bins[:-1]) / 2
                    interp = scipy.interpolate.interp1d(bins_center, hist, kind='nearest', fill_value='extrapolate', bounds_error=False)
                    effective_area_map[i] += np.maximum(hp.smoothing(interp(np.sin(np.radians(dec))) * subana.livetime, np.radians(5)), 0)

        return effective_area_map
