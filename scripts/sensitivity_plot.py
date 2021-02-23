"""
Generate a sensitivity plot for the 10-year PS analysis.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)
import healpy as hp
try:
    from tqdm import tqdm
except:
    def tqdm(arg): return arg

from KIPAC.nuXgal import Defaults
from KIPAC.nuXgal.Likelihood import Likelihood
from KIPAC.nuXgal.EventGenerator import EventGenerator
from KIPAC.nuXgal.NeutrinoSample import NeutrinoSample
from KIPAC.nuXgal.FermipyCastro import LnLFn
from KIPAC.nuXgal.GalaxySample import GALAXY_LIBRARY
from KIPAC.nuXgal.Exposure import ICECUBE_EXPOSURE_LIBRARY

def null_flux_ul(N_re):
    """
    Generate synthetic data and do a maximum likelihood estimate on the flux.
    """

    gs_WISE = GALAXY_LIBRARY.get_sample('WISE')

    #w_cross = np.zeros((N_re, Defaults.NEbin, 3 * Defaults.NSIDE))
    #Ncount_av = np.zeros(Defaults.NEbin)
    ns = NeutrinoSample()

    eg_list = [
        EventGenerator('IC40'),
        EventGenerator('IC59'),
        EventGenerator('IC79'),
        EventGenerator('IC86_I'),
        EventGenerator('IC86_II'),
        EventGenerator('IC86_III'),
        EventGenerator('IC86_IV'),
        EventGenerator('IC86_V'),
        EventGenerator('IC86_VI'),
        EventGenerator('IC86_VII')
        ]


    llh = Likelihood(10, 'WISE', computeSTD=False, Ebinmin=1, Ebinmax=4, lmin=50)
    #llh = Likelihood(10, 'WISE', computeSTD=True, Ebinmin=1, Ebinmax=4, lmin=50)

    upper_limit_flux = np.zeros((N_re, llh.Ebinmax - llh.Ebinmin))

    for i in tqdm(np.arange(N_re)):
        datamap = sum([eg.SyntheticData(1., f_diff=0, density_nu=gs_WISE.density) for eg in eg_list])
        ns = NeutrinoSample()
        ns.inputCountsmap(datamap)
        
        #write_maps_to_fits(countsmap, astropath)

        llh.inputData(ns)
        ns.updateMask(llh.idx_mask)
        bestfit_f, TS = minimizeResult = llh.minimize__lnL()


        # upper limit calculation; mostly copied from plotCastro
        # common x for castro object initialization
        f_Ebin = np.linspace(0, 4, 1000)

        exposuremap = ICECUBE_EXPOSURE_LIBRARY.get_exposure('IC86_II', 2.28)

        for idx_E in range(llh.Ebinmin, llh.Ebinmax):
            # exposuremap assuming alpha = 2.28 (numu) to convert bestfit f_astro to flux
            exposuremap_E = exposuremap[idx_E].copy()
            exposuremap_E[llh.idx_mask] = hp.UNSEEN
            exposuremap_E = hp.ma(exposuremap_E)
            factor_f2flux = llh.Ncount[idx_E] / (exposuremap_E.mean() * 1e4 * Defaults.DT_SECONDS *
                                                  llh.N_yr * 4 * np.pi * llh.f_sky * Defaults.map_dlogE *
                                                  np.log(10.)) * Defaults.map_E_center[idx_E]

            idx_bestfit_f = idx_E - llh.Ebinmin
            lnl_max = llh.log_likelihood_Ebin(bestfit_f[idx_bestfit_f], idx_E)
            lnL_Ebin = np.zeros_like(f_Ebin)
            for idx_f, f in enumerate(f_Ebin):
                lnL_Ebin[idx_f] = llh.log_likelihood_Ebin(f, idx_E)

            castro = LnLFn(f_Ebin, -lnL_Ebin)
            TS_Ebin = castro.TS()

            f_hi = castro.getLimit(0.1)
            flux_hi = f_hi * factor_f2flux
            upper_limit_flux[i, idx_E - 1] = flux_hi

    plot_upper_limit_histogram(upper_limit_flux)
    return np.median(upper_limit_flux, axis=0)

def plot_upper_limit_histogram(upper_limits):
    fig, ax = plt.subplots(1, 3, figsize=(12, 8))
    ax = ax.flatten()
    for i in np.arange(upper_limits.shape[1]):
        ax[i].hist(upper_limits[:,i], bins=10)
        ax[i].set_title('Energy bin {i}'.format(i=i))
        ax[i].set_xlabel(r'$E^2 \frac{dN}{dE}$ [GeV/cm$^2$/sec/sr]')
        ax[i].axvline(np.median(upper_limits[:,i]), c='black')


    fig.savefig(os.path.join(Defaults.NUXGAL_PLOT_DIR, 'upper_limits_dist.pdf'))

def plot_sensitivity(sens_limit, det_limit):
    llh = Likelihood(10, 'WISE', computeSTD=False, Ebinmin=1, Ebinmax=4, lmin=50)

    fig, ax = plt.subplots()

    idx_E = np.arange(llh.Ebinmin, llh.Ebinmax)
    E = Defaults.map_logE_center[idx_E]

    ax.errorbar(E, sens_limit, xerr=Defaults.map_dlogE/2., fmt='.', label='Sensitivity limit')

    ax.set_xlabel(r'$log_{10}(E/GeV)$')
    ax.set_ylabel(r'$E^2 \frac{dE}{dN}$ [GeV/cm$^2$/sec/sr]')
    ax.set_xticks(E)
    ax.grid(True, linestyle='--', which='both')

    ax.legend()
    ax.semilogy()

    #ax.scatter(E, det_limit) TODO

    fig.savefig(os.path.join(Defaults.NUXGAL_PLOT_DIR, 'sensitivity_plot.pdf'))


def main():

    # Calculate a test statistic distribution for flux=0.
    upper_lim = null_flux_ul(1000)

    # calculate the 5-sigma detection threshold
    plot_sensitivity(upper_lim, [0, 0, 0])


if __name__ == "__main__":
    main()

