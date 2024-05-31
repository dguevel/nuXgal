"""Contains GalaxySample class to organize galaxy samples 
for cross correlation"""


import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import healpy as hp
import os
import tempfile
try:
    from ispice import ispice
except ImportError:
    print("PolSpice not installed, experimental PolSpice cross correlation will raise error")

import matplotlib.pyplot as plt

from . import Defaults


class GalaxySample():
    """Class to organize galaxy samples for cross correlation
    """
    def __init__(self, galaxyName, idx_galaxymask):
        """C'tor

        Currently this implements:
        flat : Poisson generated flat galaxy sample

        Parameters
        ----------
        galaxyName : `str`
            Name for the sample, used to define the sample and specify output file paths
        idx_galaxymask :
            Used to ma

        """
        self.galaxyName = galaxyName
        self.idx_galaxymask = idx_galaxymask
        self.f_sky = 1. - len(self.idx_galaxymask[0]) / float(Defaults.NPIXEL)
        galaxymap_path = Defaults.GALAXYMAP_FORMAT.format(galaxyName=galaxyName)
        #overdensityalm_path = Defaults.GALAXYALM_FORMAT.format(galaxyName=galaxyName)
        self.galaxymap = hp.fitsfunc.read_map(galaxymap_path)
        self.galaxymap[self.idx_galaxymask] = hp.UNSEEN
        self.galaxymap = hp.ma(self.galaxymap)
        self.overdensity = self.galaxymap / self.galaxymap.mean() - 1.
        self.overdensityalm = hp.map2alm(self.overdensity, lmax=Defaults.MAX_L) / self.f_sky
        self.density = self.galaxymap / np.sum(self.galaxymap)


    def plotGalaxymap(self, plotmax=100):
        """Plot galaxy counts map for a particular sample

        Parameters
        ----------
        plotmax : `float`
            Maximum value in the plot
        """
        hp.mollview(self.galaxymap, title=self.galaxyName, max=plotmax)
        testfigpath = Defaults.GALAXYMAP_FIG_FORMAT.format(
            galaxyName=self.galaxyName)
        plt.savefig(testfigpath)

    def getAutoCorrelation(self):
        """Return the auto correlation of the galaxy sample"""

        w_auto = hp.anafast(self.overdensity, lmax=Defaults.MAX_L) / self.f_sky
        return w_auto

    def getAutoCorrelationPolSpice(self):
        """Return the auto correlation of the galaxy sample"""

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

            hp.write_map(galaxy_map_fname, self.overdensity)
            galaxy_mask = np.zeros_like(self.overdensity, dtype=bool)
            galaxy_mask[self.mask()] = True
            galaxy_mask = ~galaxy_mask
            hp.write_map(galaxy_mask_fname, galaxy_mask, dtype=int)

            cl_out_fname = os.path.join(
                temp_dir,
                'cl_out.fits'
            )

            # run PolSpice
            ispice(
                mapin1=galaxy_map_fname,
                maskfile1=galaxy_mask_fname,
                clout=cl_out_fname,
                apodizesigma=180,
                thetamax=180,
                subav=True,
                subdipole=True,
            )

            # read the output and load into the cross correlation array
            w = hp.read_cl(cl_out_fname)

        return w

class GalaxySample_Atmospheric(GalaxySample):
    @staticmethod
    def mask():
        """Contstruct and return the mask for this sample"""
        c_icrs = SkyCoord(ra=Defaults.exposuremap_phi * u.radian,
                          dec=(np.pi/2 - Defaults.exposuremap_theta)*u.radian, frame='icrs')
        return np.where((np.abs(c_icrs.galactic.b.degree) < 10) | (c_icrs.dec.degree < -5))

    def __init__(self):
        """C'tor"""
        GalaxySample.__init__(self, "Atmospheric", self.mask())


class GalaxySample_SDSS_00_01(GalaxySample):
    """SDSS galaxy sample

    SDSS galaxy sample map based on ~1M galaxies
    """
    @staticmethod
    def mask():
        """Contstruct and return the mask for this sample"""
        c_icrs = SkyCoord(ra=Defaults.exposuremap_phi * u.radian,
                          dec=(np.pi/2 - Defaults.exposuremap_theta)*u.radian, frame='icrs')
        sdss_footprint = hp.read_map('/home/dguevel/git/nuXgal/data/ancil/SDSS_z0.0_0.1_mask.fits')
        return np.where((np.abs(c_icrs.galactic.b.degree) < 10) | (sdss_footprint == 0))

    def __init__(self):
        """C'tor"""
        self.z = 0.05
        self.dz = 0.1
        GalaxySample.__init__(self, "SDSS_z0.0_0.1", self.mask())


class GalaxySample_SDSS_01_02(GalaxySample):
    """SDSS galaxy sample

    SDSS galaxy sample map based on ~1M galaxies
    """
    @staticmethod
    def mask():
        """Contstruct and return the mask for this sample"""
        c_icrs = SkyCoord(ra=Defaults.exposuremap_phi * u.radian,
                          dec=(np.pi/2 - Defaults.exposuremap_theta)*u.radian, frame='icrs')
        sdss_footprint = hp.read_map('/home/dguevel/git/nuXgal/data/ancil/SDSS_z0.1_0.2_mask.fits')
        return np.where((np.abs(c_icrs.galactic.b.degree) < 10) | (sdss_footprint == 0))

    def __init__(self):
        """C'tor"""
        self.z = 0.15
        self.dz = 0.1
        GalaxySample.__init__(self, "SDSS_z0.1_0.2", self.mask())


class GalaxySample_SDSS_02_03(GalaxySample):
    """SDSS galaxy sample

    SDSS galaxy sample map based on ~1M galaxies
    """
    @staticmethod
    def mask():
        """Contstruct and return the mask for this sample"""
        c_icrs = SkyCoord(ra=Defaults.exposuremap_phi * u.radian,
                          dec=(np.pi/2 - Defaults.exposuremap_theta)*u.radian, frame='icrs')
        sdss_footprint = hp.read_map('/home/dguevel/git/nuXgal/data/ancil/SDSS_z0.2_0.3_mask.fits')
        return np.where((np.abs(c_icrs.galactic.b.degree) < 10) | (sdss_footprint == 0))

    def __init__(self):
        """C'tor"""
        self.z = 0.25
        self.dz = 0.1
        GalaxySample.__init__(self, "SDSS_z0.2_0.3", self.mask())


class GalaxySample_SDSS_03_04(GalaxySample):
    """SDSS galaxy sample

    SDSS galaxy sample map based on ~1M galaxies
    """
    @staticmethod
    def mask():
        """Contstruct and return the mask for this sample"""
        c_icrs = SkyCoord(ra=Defaults.exposuremap_phi * u.radian,
                          dec=(np.pi/2 - Defaults.exposuremap_theta)*u.radian, frame='icrs')
        sdss_footprint = hp.read_map('/home/dguevel/git/nuXgal/data/ancil/SDSS_z0.3_0.4_mask.fits')
        return np.where((np.abs(c_icrs.galactic.b.degree) < 10) | (sdss_footprint == 0))

    def __init__(self):
        """C'tor"""
        self.z = 0.35
        self.dz = 0.1
        GalaxySample.__init__(self, "SDSS_z0.3_0.4", self.mask())


class GalaxySample_SDSS_04_05(GalaxySample):
    """SDSS galaxy sample

    SDSS galaxy sample map based on ~1M galaxies
    """
    @staticmethod
    def mask():
        """Contstruct and return the mask for this sample"""
        c_icrs = SkyCoord(ra=Defaults.exposuremap_phi * u.radian,
                          dec=(np.pi/2 - Defaults.exposuremap_theta)*u.radian, frame='icrs')
        sdss_footprint = hp.read_map('/home/dguevel/git/nuXgal/data/ancil/SDSS_z0.4_0.5_mask.fits')
        return np.where((np.abs(c_icrs.galactic.b.degree) < 10) | (sdss_footprint == 0))

    def __init__(self):
        """C'tor"""
        self.z = 0.45
        self.dz = 0.1
        GalaxySample.__init__(self, "SDSS_z0.4_0.5", self.mask())


class GalaxySample_SDSS_05_06(GalaxySample):
    """SDSS galaxy sample

    SDSS galaxy sample map based on ~1M galaxies
    """
    @staticmethod
    def mask():
        """Contstruct and return the mask for this sample"""
        c_icrs = SkyCoord(ra=Defaults.exposuremap_phi * u.radian,
                          dec=(np.pi/2 - Defaults.exposuremap_theta)*u.radian, frame='icrs')
        sdss_footprint = hp.read_map('/home/dguevel/git/nuXgal/data/ancil/SDSS_z0.5_0.6_mask.fits')
        return np.where((np.abs(c_icrs.galactic.b.degree) < 10) | (sdss_footprint == 0))

    def __init__(self):
        """C'tor"""
        self.z = 0.55
        self.dz = 0.1
        GalaxySample.__init__(self, "SDSS_z0.5_0.6", self.mask())


class GalaxySample_SDSS_06_07(GalaxySample):
    """SDSS galaxy sample

    SDSS galaxy sample map based on ~1M galaxies
    """
    @staticmethod
    def mask():
        """Contstruct and return the mask for this sample"""
        c_icrs = SkyCoord(ra=Defaults.exposuremap_phi * u.radian,
                          dec=(np.pi/2 - Defaults.exposuremap_theta)*u.radian, frame='icrs')
        sdss_footprint = hp.read_map('/home/dguevel/git/nuXgal/data/ancil/SDSS_z0.6_0.7_mask.fits')
        return np.where((np.abs(c_icrs.galactic.b.degree) < 10) | (sdss_footprint == 0))

    def __init__(self):
        """C'tor"""
        self.z = 0.65
        self.dz = 0.1
        GalaxySample.__init__(self, "SDSS_z0.6_0.7", self.mask())


class GalaxySample_SDSS_07_08(GalaxySample):
    """SDSS galaxy sample

    SDSS galaxy sample map based on ~1M galaxies
    """
    @staticmethod
    def mask():
        """Contstruct and return the mask for this sample"""
        c_icrs = SkyCoord(ra=Defaults.exposuremap_phi * u.radian,
                          dec=(np.pi/2 - Defaults.exposuremap_theta)*u.radian, frame='icrs')
        sdss_footprint = hp.read_map('/home/dguevel/git/nuXgal/data/ancil/SDSS_z0.7_0.8_mask.fits')
        return np.where((np.abs(c_icrs.galactic.b.degree) < 10) | (sdss_footprint == 0))

    def __init__(self):
        """C'tor"""
        self.z = 0.75
        self.dz = 0.1
        GalaxySample.__init__(self, "SDSS_z0.7_0.8", self.mask())


class GalaxySample_SDSS_08_10(GalaxySample):
    """SDSS galaxy sample

    SDSS galaxy sample map based on ~1M galaxies
    """
    @staticmethod
    def mask():
        """Contstruct and return the mask for this sample"""
        c_icrs = SkyCoord(ra=Defaults.exposuremap_phi * u.radian,
                          dec=(np.pi/2 - Defaults.exposuremap_theta)*u.radian, frame='icrs')
        sdss_footprint = hp.read_map('/home/dguevel/git/nuXgal/data/ancil/SDSS_z0.8_1.0_mask.fits')
        return np.where((np.abs(c_icrs.galactic.b.degree) < 10) | (sdss_footprint == 0))

    def __init__(self):
        """C'tor"""
        self.z = 0.9
        self.dz = 0.2
        GalaxySample.__init__(self, "SDSS_z0.8_1.0", self.mask())


class GalaxySample_SDSS_10_12(GalaxySample):
    """SDSS galaxy sample

    SDSS galaxy sample map based on ~1M galaxies
    """
    @staticmethod
    def mask():
        """Contstruct and return the mask for this sample"""
        c_icrs = SkyCoord(ra=Defaults.exposuremap_phi * u.radian,
                          dec=(np.pi/2 - Defaults.exposuremap_theta)*u.radian, frame='icrs')
        sdss_footprint = hp.read_map('/home/dguevel/git/nuXgal/data/ancil/SDSS_z1.0_1.2_mask.fits')
        return np.where((np.abs(c_icrs.galactic.b.degree) < 10) | (sdss_footprint == 0))

    def __init__(self):
        """C'tor"""
        self.z = 1.1
        self.dz = 0.2
        GalaxySample.__init__(self, "SDSS_z1.0_1.2", self.mask())


class GalaxySample_SDSS_12_14(GalaxySample):
    """SDSS galaxy sample

    SDSS galaxy sample map based on ~1M galaxies
    """
    @staticmethod
    def mask():
        """Contstruct and return the mask for this sample"""
        c_icrs = SkyCoord(ra=Defaults.exposuremap_phi * u.radian,
                          dec=(np.pi/2 - Defaults.exposuremap_theta)*u.radian, frame='icrs')
        sdss_footprint = hp.read_map('/home/dguevel/git/nuXgal/data/ancil/SDSS_z1.2_1.4_mask.fits')
        return np.where((np.abs(c_icrs.galactic.b.degree) < 10) | (sdss_footprint == 0))

    def __init__(self):
        """C'tor"""
        self.z = 1.3
        self.dz = 0.2
        GalaxySample.__init__(self, "SDSS_z1.2_1.4", self.mask())


class GalaxySample_SDSS_14_16(GalaxySample):
    """SDSS galaxy sample

    SDSS galaxy sample map based on ~1M galaxies
    """
    @staticmethod
    def mask():
        """Contstruct and return the mask for this sample"""
        c_icrs = SkyCoord(ra=Defaults.exposuremap_phi * u.radian,
                          dec=(np.pi/2 - Defaults.exposuremap_theta)*u.radian, frame='icrs')
        sdss_footprint = hp.read_map('/home/dguevel/git/nuXgal/data/ancil/SDSS_z1.4_1.6_mask.fits')
        return np.where((np.abs(c_icrs.galactic.b.degree) < 10) | (sdss_footprint == 0))

    def __init__(self):
        """C'tor"""
        self.z = 1.5
        self.dz = 0.2
        GalaxySample.__init__(self, "SDSS_z1.4_1.6", self.mask())


class GalaxySample_SDSS_16_18(GalaxySample):
    """SDSS galaxy sample

    SDSS galaxy sample map based on ~1M galaxies
    """
    @staticmethod
    def mask():
        """Contstruct and return the mask for this sample"""
        c_icrs = SkyCoord(ra=Defaults.exposuremap_phi * u.radian,
                          dec=(np.pi/2 - Defaults.exposuremap_theta)*u.radian, frame='icrs')
        sdss_footprint = hp.read_map('/home/dguevel/git/nuXgal/data/ancil/SDSS_z1.6_1.8_mask.fits')
        return np.where((np.abs(c_icrs.galactic.b.degree) < 10) | (sdss_footprint == 0))

    def __init__(self):
        """C'tor"""
        self.z = 1.7
        self.dz = 0.2
        GalaxySample.__init__(self, "SDSS_z1.6_1.8", self.mask())


class GalaxySample_SDSS_18_20(GalaxySample):
    """SDSS galaxy sample

    SDSS galaxy sample map based on ~1M galaxies
    """
    @staticmethod
    def mask():
        """Contstruct and return the mask for this sample"""
        c_icrs = SkyCoord(ra=Defaults.exposuremap_phi * u.radian,
                          dec=(np.pi/2 - Defaults.exposuremap_theta)*u.radian, frame='icrs')
        sdss_footprint = hp.read_map('/home/dguevel/git/nuXgal/data/ancil/SDSS_z1.8_2.0_mask.fits')
        return np.where((np.abs(c_icrs.galactic.b.degree) < 10) | (sdss_footprint == 0))

    def __init__(self):
        """C'tor"""
        self.z = 1.9
        self.dz = 0.2
        GalaxySample.__init__(self, "SDSS_z1.8_2.0", self.mask())

class GalaxySample_SDSS_20_22(GalaxySample):
    """SDSS galaxy sample

    SDSS galaxy sample map based on ~1M galaxies
    """
    @staticmethod
    def mask():
        """Contstruct and return the mask for this sample"""
        c_icrs = SkyCoord(ra=Defaults.exposuremap_phi * u.radian,
                          dec=(np.pi/2 - Defaults.exposuremap_theta)*u.radian, frame='icrs')
        sdss_footprint = hp.read_map('/home/dguevel/git/nuXgal/data/ancil/SDSS_z2.0_2.2_mask.fits')
        return np.where((np.abs(c_icrs.galactic.b.degree) < 10) | (sdss_footprint == 0))

    def __init__(self):
        """C'tor"""
        self.z = 2.1
        self.dz = 0.2
        GalaxySample.__init__(self, "SDSS_z2.0_2.2", self.mask())


class GalaxySample_Planck(GalaxySample):
    """Planck cosmic infrared background"""
    @staticmethod
    def mask():
        """Contstruct and return the mask for this sample"""
        c_icrs = SkyCoord(ra=Defaults.exposuremap_phi * u.radian,
                          dec=(np.pi/2 - Defaults.exposuremap_theta)*u.radian, frame='icrs')
        return np.where(np.abs(c_icrs.galactic.b.degree) < 10)

    def __init__(self):
        """C'tor"""
        GalaxySample.__init__(self, "Planck", self.mask())


class GalaxySample_Wise(GalaxySample):
    """WISE Galaxy sample

    WISE-2MASS galaxy sample map based on ~5M galaixes
    """
    @staticmethod
    def mask():
        """Contstruct and return the mask for this sample"""
        c_icrs = SkyCoord(ra=Defaults.exposuremap_phi * u.radian,
                          dec=(np.pi/2 - Defaults.exposuremap_theta)*u.radian, frame='icrs')
        return np.where(np.abs(c_icrs.galactic.b.degree) < 10)

    def __init__(self):
        """C'tor"""
        GalaxySample.__init__(self, "WISE", self.mask())


class GalaxySample_unWise_z04(GalaxySample):
    """WISE Galaxy sample

    WISE-2MASS galaxy sample map based on ~5M galaixes
    """
    @staticmethod
    def mask():
        """Contstruct and return the mask for this sample"""
        c_icrs = SkyCoord(ra=Defaults.exposuremap_phi * u.radian,
                          dec=(np.pi/2 - Defaults.exposuremap_theta)*u.radian, frame='icrs')
        
        # planck dust map
        planck_dustmap = hp.read_map('/home/dguevel/git/nuXgal/data/ancil/HFI_Mask_GalPlane-apo0_2048_R2.00_nside128.fits')

        # large and small magellanic clouds
        lmc = hp.query_disc(Defaults.NSIDE, hp.ang2vec(80.894200, -69.756100, lonlat=True), 5*np.pi/180)
        smc = hp.query_disc(Defaults.NSIDE, hp.ang2vec(13.158300, -72.800300, lonlat=True), 5*np.pi/180)
        mc = np.zeros(Defaults.NPIXEL)
        mc[lmc] = 1
        mc[smc] = 1
        return np.where((np.abs(c_icrs.galactic.b.degree) < 10) | (planck_dustmap == 0) | (mc == 1))

    def __init__(self):
        """C'tor"""
        GalaxySample.__init__(self, "unWISE_z=0.4", self.mask())


class GalaxySample_unWise_z06(GalaxySample):
    """WISE Galaxy sample

    WISE-2MASS galaxy sample map based on ~5M galaixes
    """
    @staticmethod
    def mask():
        """Contstruct and return the mask for this sample"""
        c_icrs = SkyCoord(ra=Defaults.exposuremap_phi * u.radian,
                          dec=(np.pi/2 - Defaults.exposuremap_theta)*u.radian, frame='icrs')
        return np.where(np.abs(c_icrs.galactic.b.degree) < 10)

    def __init__(self):
        """C'tor"""
        GalaxySample.__init__(self, "unWISE_z=0.6", self.mask())


class GalaxySample_unWise_z10(GalaxySample):
    """WISE Galaxy sample

    WISE-2MASS galaxy sample map based on ~5M galaixes
    """
    @staticmethod
    def mask():
        """Contstruct and return the mask for this sample"""
        c_icrs = SkyCoord(ra=Defaults.exposuremap_phi * u.radian,
                          dec=(np.pi/2 - Defaults.exposuremap_theta)*u.radian, frame='icrs')
        return np.where(np.abs(c_icrs.galactic.b.degree) < 10)

    def __init__(self):
        """C'tor"""
        GalaxySample.__init__(self, "unWISE_z=1.0", self.mask())


class GalaxySample_unWise_z15(GalaxySample):
    """WISE Galaxy sample

    WISE-2MASS galaxy sample map based on ~5M galaixes
    """
    @staticmethod
    def mask():
        """Contstruct and return the mask for this sample"""
        c_icrs = SkyCoord(ra=Defaults.exposuremap_phi * u.radian,
                          dec=(np.pi/2 - Defaults.exposuremap_theta)*u.radian, frame='icrs')
        return np.where(np.abs(c_icrs.galactic.b.degree) < 10)

    def __init__(self):
        """C'tor"""
        GalaxySample.__init__(self, "unWISE_z=1.5", self.mask())


class GalaxySample_Analy(GalaxySample):
    """Galaxy sample from analytic CL

    Simulated galaxy sample based on analytical power spectrum
    """
    @staticmethod
    def mask():
        """Contstruct and return the mask for this sample"""
        return np.where(False)

    def __init__(self):
        """C'tor"""
        GalaxySample.__init__(self, "analy", self.mask())
        self.analyCL = np.loadtxt(Defaults.ANALYTIC_CL_PATH)


class GalaxySample_Flat(GalaxySample):
    """Flat galaxy sample

    Simulated flat galaxy sample
    """
    def __init__(self):
        """C'tor

        Currently this implements:
        flat : Poisson generated flat galaxy sample

        Parameters
        ----------
        galaxyName : `str`
            Name for the sample, used to define the sample and specify output file paths
        idx_galaxymask :
            Used to ma

        """
        self.galaxyName = 'flat'
        self.idx_galaxymask = self.mask()
        self.f_sky = 1. - len(self.idx_galaxymask[0]) / float(Defaults.NPIXEL)
        #overdensityalm_path = Defaults.GALAXYALM_FORMAT.format(galaxyName=galaxyName)
        self.galaxymap = np.ones((Defaults.NPIXEL), float) / Defaults.NPIXEL
        self.galaxymap[self.idx_galaxymask] = hp.UNSEEN
        self.galaxymap = hp.ma(self.galaxymap)
        self.overdensity = self.galaxymap / self.galaxymap.mean() - 1.
        self.overdensityalm = hp.map2alm(self.overdensity, lmax=Defaults.MAX_L) / self.f_sky
        self.density = self.galaxymap / np.sum(self.galaxymap)

    @staticmethod
    def mask():
        """Contstruct and return the mask for this sample"""
        return np.where(np.zeros(Defaults.NPIXEL, dtype=bool))


class GalaxySampleLibrary:
    """Library of galaxy samples"""

    galaxy_class_dict = {
        'WISE': GalaxySample_Wise,
        'analy': GalaxySample_Analy,
        'flat': GalaxySample_Flat,
        'Planck': GalaxySample_Planck,
        'unWISE_z=0.4': GalaxySample_unWise_z04,
        'unWISE_z=0.6': GalaxySample_unWise_z06,
        'unWISE_z=1.0': GalaxySample_unWise_z10,
        'unWISE_z=1.5': GalaxySample_unWise_z15,
        'Atmospheric': GalaxySample_Atmospheric,
        'SDSS_z0.0_0.1': GalaxySample_SDSS_00_01,
        'SDSS_z0.1_0.2': GalaxySample_SDSS_01_02,
        'SDSS_z0.2_0.3': GalaxySample_SDSS_02_03,
        'SDSS_z0.3_0.4': GalaxySample_SDSS_03_04,
        'SDSS_z0.4_0.5': GalaxySample_SDSS_04_05,
        'SDSS_z0.5_0.6': GalaxySample_SDSS_05_06,
        'SDSS_z0.6_0.7': GalaxySample_SDSS_06_07,
        'SDSS_z0.7_0.8': GalaxySample_SDSS_07_08,
        'SDSS_z0.8_1.0': GalaxySample_SDSS_08_10,
        'SDSS_z1.0_1.2': GalaxySample_SDSS_10_12,
        'SDSS_z1.2_1.4': GalaxySample_SDSS_12_14,
        'SDSS_z1.4_1.6': GalaxySample_SDSS_14_16,
        'SDSS_z1.6_1.8': GalaxySample_SDSS_16_18,
        'SDSS_z1.8_2.0': GalaxySample_SDSS_18_20,
        'SDSS_z2.0_2.2': GalaxySample_SDSS_20_22}

    def __init__(self, randomseed_galaxy=Defaults.randomseed_galaxy):
        """C'tor"""
        self._gs_dict = {}
        self.randomseed_galaxy = randomseed_galaxy

    def keys(self):
        """Return the names of exposure maps"""
        return self._gs_dict.keys()

    def values(self):
        """Returns the exposure maps"""
        return self._gs_dict.values()

    def items(self):
        """Return the name : map pairs"""
        return self._gs_dict.items()

    def __getitem__(self, key):
        """Return a particular exposure map by name"""
        return self._gs_dict[key]

    def get_sample(self, sampleName):
        """Get a particular sample by name"""
        try:
            return self.galaxy_class_dict[sampleName]()
        except KeyError:
            print("Galaxy Sample %s not defined, options are %s" % (sampleName, str(self._gs_dict.keys())))
        return None

    def generateGalaxy(self, N_g=2000000, write_map=True):
        """Generate a synthetic galaxy sample

        Parameters
        ----------
        N_g : `int`
            Number of galaxies to generate
        write_map: `bool`
            if True write the generate map to the ancilary data area
        """
        analyCL = np.loadtxt(Defaults.ANALYTIC_CL_PATH)
        np.random.seed(self.randomseed_galaxy)
        alm = hp.sphtfunc.synalm(analyCL, lmax=Defaults.MAX_L)
        density_g = hp.sphtfunc.alm2map(alm, Defaults.NSIDE)
        #density_g = hp.sphtfunc.synfast(analyCL, Defaults.NSIDE)
        density_g = np.exp(density_g)
        density_g /= density_g.sum()
        expected_counts_map = density_g * N_g
        np.random.seed(Defaults.randomseed_galaxy)

        analy_galaxymap = np.random.poisson(expected_counts_map)
        if write_map:
            analy_galaxymap_path = Defaults.GALAXYMAP_FORMAT.format(galaxyName='analy')
            hp.fitsfunc.write_map(analy_galaxymap_path, analy_galaxymap, overwrite=True)
            overdensityalm_path = Defaults.GALAXYALM_FORMAT.format(galaxyName='analy')
            hp.fitsfunc.write_alm(overdensityalm_path, alm, overwrite=True)

    def generateFlat(self, N_g=2000000, write_map=True):
        """Generate a synthetic galaxy sample

        Parameters
        ----------
        N_g : `int`
            Number of galaxies to generate
        write_map: `bool`
            if True write the generate map to the ancilary data area
        """
        np.random.seed(self.randomseed_galaxy)
        density_g = np.ones((Defaults.NPIXEL), float)
        density_g /= density_g.sum()
        expected_counts_map = density_g * N_g
        np.random.seed(Defaults.randomseed_galaxy)

        analy_galaxymap = np.random.poisson(expected_counts_map)
        if write_map:
            analy_galaxymap_path = Defaults.GALAXYMAP_FORMAT.format(galaxyName='flat')
            hp.fitsfunc.write_map(analy_galaxymap_path, analy_galaxymap, overwrite=True)
            overdensityalm_path = Defaults.GALAXYALM_FORMAT.format(galaxyName='flat')
            hp.fitsfunc.write_alm(overdensityalm_path, alm, overwrite=True)


GALAXY_LIBRARY = GalaxySampleLibrary(Defaults.randomseed_galaxy)
