import os
import argparse
import numpy as np
import healpy as hp

from KIPAC.nuXgal import Defaults
from KIPAC.nuXgal import file_utils
from KIPAC.nuXgal import FigureDict
from KIPAC.nuXgal.Exposure import WeightedAeff

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", default=None, required=True,
                    help="Directory with IceCube data")

parser.add_argument("-o", "--output", default=Defaults.NUXGAL_DIR,
                    help="Output directory")

args = parser.parse_args()

icecube_data_dir = os.path.join(args.input, 'data/icecube_10year_ps/events')
data_dir = os.path.join(args.output, 'data', 'data')
irf_dir = os.path.join(args.output, 'data', 'irfs')
plot_dir = os.path.join(args.output, 'plots')
testfigpath = os.path.join(Defaults.NUXGAL_PLOT_DIR, 'test')
countsmap3yr_format = os.path.join(Defaults.NUXGAL_DATA_DIR,  'IceCube3yr_countsmap{i}.fits')

for dirname in [data_dir, irf_dir, plot_dir]:
    try:
        os.makedirs(dirname)
    except OSError:
        pass

def geneateNcos_thetaFile(year):

    Nzenith_bin = 100
    N_coszenith = np.zeros((len(Defaults.map_logE_center), Nzenith_bin))

    for file in [os.path.join(icecube_data_dir, year + '_exp.csv')]:

        AtmBG_file = np.loadtxt(file)
        _index_map_logE = np.searchsorted(Defaults.map_logE_edge, AtmBG_file[:, 1]) - 1
        _index_map_logE[_index_map_logE == -1] = 0 # group logE < 2 events to bin 0

        for i in np.arange(len(Defaults.map_logE_center)):
            """ use healpy convention: theta = 180 - zenith_IceCube """
            N_coszenith_i, cosZenithBinEdges =\
                np.histogram(np.cos(np.pi - AtmBG_file[:, 6][_index_map_logE == i] * np.pi / 180.), Nzenith_bin, (-1, 1))
            N_coszenith[i] += N_coszenith_i


    cosZenithBinCenters = (cosZenithBinEdges[0:-1] + cosZenithBinEdges[1:])/2.

    figs = FigureDict()
    figs.plot_yvals('N_coszenith', cosZenithBinCenters, N_coszenith,
                    xlabel=r'$\cos\,\theta$',
                    ylabel='Number of counts in 2010-2012 data',
                    figsize=(8, 6))

    for i in np.arange(Defaults.NEbin):
        np.savetxt(os.path.join(irf_dir, 'Ncos_theta_'+year+'_'+str(i)+'.txt'),
                   np.column_stack((cosZenithBinCenters, N_coszenith[i])))


def generateCountsmap(year):
    countsmap = np.zeros((Defaults.NEbin, hp.pixelfunc.nside2npix(Defaults.NSIDE)))
    AtmBG_file = np.loadtxt(os.path.join(icecube_data_dir, year+'_exp.csv'))

    # get energy index of events
    _index_map_logE = np.searchsorted(Defaults.map_logE_edge, AtmBG_file[:, 1]) - 1
    _index_map_logE[_index_map_logE == -1] = 0 # group logE < 2 events to bin 0 (we don't use bin 0)

    # convert event directions to pixel numbers
    # pi/2 - dec = theta, ra = phi
    _index_map_pixel = hp.pixelfunc.ang2pix(Defaults.NSIDE, theta=np.radians(90. - AtmBG_file[:, 4]) , phi=np.radians(AtmBG_file[:, 3]))

    # put events into healpy maps
    for i, _ in enumerate(AtmBG_file):
        countsmap[_index_map_logE[i]][_index_map_pixel[i]] += 1.

    return countsmap

if __name__ == '__main__':
    countsmap = np.zeros((Defaults.NEbin, hp.pixelfunc.nside2npix(Defaults.NSIDE)))
    for year in ['IC40', 'IC59', 'IC79', 'IC86_I', 'IC86_II', 'IC86_III', 'IC86_IV', 'IC86_V', 'IC86_VI', 'IC86_VII']:
        geneateNcos_thetaFile(year)
        countsmap = countsmap + generateCountsmap(year)

        for spectralIndex in [3.7, 2.28, 2.89]:
            WeightedAeff(year=year, spectralIndex=spectralIndex)

    file_utils.write_maps_to_fits(countsmap, countsmap3yr_format)
