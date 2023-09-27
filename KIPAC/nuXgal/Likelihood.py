"""Defintion of Likelihood function"""

import os
import numpy as np
import healpy as hp
import emcee
import corner
import csky as cy
import json
import itertools
import warnings
from multiprocessing import Pool


from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from pandas import concat

from scipy.optimize import minimize
from scipy.stats import norm, distributions
from scipy.interpolate import interp1d

from .EventGenerator import EventGenerator
from . import Defaults
from .NeutrinoSample import NeutrinoSample
from .FermipyCastro import LnLFn
from .GalaxySample import GALAXY_LIBRARY
from .Exposure import ICECUBE_EXPOSURE_LIBRARY
from .CskyEventGenerator import CskyEventGenerator
from .Models import DataScrambleBackgroundModel


def significance(chi_square, dof):
    """Construct an significance for a chi**2 distribution

    Parameters
    ----------
    chi_square : `float`
    dof : `int`

    Returns
    -------
    significance : `float`
    """
    p_value = distributions.chi2.sf(chi_square, dof)
    significance_twoTailNorm = norm.isf(p_value/2)
    return significance_twoTailNorm


def significance_from_chi(chi):
    """Construct an significance set of chi values

    Parameters
    ----------
    chi : `array`
    dof : `int`

    Returns
    -------
    significance : `float`
    """
    chi2 = chi*chi
    dof = len(chi2)
    return significance(np.sum(chi2), dof)


class Likelihood():
    """Class to evaluate the likelihood for a particular model of neutrino
    galaxy correlation"""
    WMeanFname = Defaults.W_MEAN_FORMAT
    AtmSTDFname = Defaults.SYNTHETIC_ATM_CROSS_CORR_STD_FORMAT
    AtmNcountsFname = Defaults.SYNTHETIC_ATM_NCOUNTS_FORMAT
    AtmMeanFname = Defaults.SYNTHETIC_ATM_W_MEAN_FORMAT
    AstroMeanFname = Defaults.SYNTHETIC_ASTRO_W_MEAN_FORMAT
    AstroSTDFname = Defaults.SYNTHETIC_ASTRO_W_STD_FORMAT
    neutrino_sample_class = NeutrinoSample

    def __init__(self, N_yr, galaxyName, Ebinmin, Ebinmax, lmin, gamma=2.5, recompute_model=False):
        """C'tor

        Parameters
        ----------
        N_yr : `float`
            Number of years to simulate if computing the models
        galaxyName : `str`
            Name of the Galaxy sample
        computeSTD : `bool`
            If true, compute the standard deviation for a number of trials
        Ebinmin, Ebinmax : `int`
           indices of energy bins between which likelihood is computed
        lmin:
            minimum of l to be taken into account in likelihood
        """

        self.N_yr = N_yr
        self.gs = GALAXY_LIBRARY.get_sample(galaxyName)
        self.anafastMask()
        self.Ebinmin = Ebinmin
        self.Ebinmax = Ebinmax
        self.lmin = lmin
        self._event_generator = None
        self.w_data = None
        self.Ncount = None
        self.gamma = gamma

        self.background_model = DataScrambleBackgroundModel(
            self.gs,
            self.N_yr,
            self.idx_mask,
            recompute=recompute_model
        )

        self.w_atm_mean = self.background_model.w_mean
        self.w_model_f1 = self.gs.getAutoCorrelation()
        self.w_atm_std = self.background_model.w_std
        self.w_atm_std_square = self.w_atm_std ** 2

    @property
    def event_generator(self):
        if self._event_generator is None:
            self._event_generator = CskyEventGenerator(
                self.N_yr,
                self.gs,
                gamma=self.gamma,
                Ebinmin=self.Ebinmin,
                Ebinmax=self.Ebinmax,
                idx_mask=self.idx_mask)
        return self._event_generator

    @staticmethod
    def init_from_run(**kwargs):
        """
        Initialize a likelihood from JSON trial output
        
        Parameters
        ----------
        **kwargs : dict
            Keyword arguments from JSON trial from nuXgal.py

        Returns
        -------
        llh : Likelihood
            Initialized Likelihood object
        """
        llh = Likelihood(
            kwargs['N_yr'],
            kwargs['galaxy_catalog'],
            kwargs['ebinmin'],
            kwargs['ebinmax'],
            kwargs['lmin'],
            gamma=kwargs['gamma'])

        llh.w_data = np.zeros((Defaults.NEbin, Defaults.NCL))
        llh.w_std = np.zeros((Defaults.NEbin, Defaults.NCL))
        for i, ebin in enumerate(range(llh.Ebinmin, llh.Ebinmax)):
            if isinstance(list(kwargs['cls'].keys())[i], str):
                ebin = str(ebin)
            llh.w_data[i] = kwargs['cls'][ebin]
            llh.w_std[i] = kwargs['cls_std'][ebin]

        return llh

    def anafastMask(self):
        """Generate a mask that merges the neutrino selection mask
        with the galaxy sample mask
        """
        # mask Southern sky to avoid muons
        mask_nu = np.zeros(Defaults.NPIXEL, dtype=bool)
        mask_nu[Defaults.idx_muon] = 1.
        # add the mask of galaxy sample
        mask_nu[self.gs.idx_galaxymask] = 1.
        self.idx_mask = np.where(mask_nu != 0)
        self.f_sky = 1. - len(self.idx_mask[0]) / float(Defaults.NPIXEL)

    def bootstrapSigma(self, ebin, niter=100, mp_cpus=1):
        cl = np.zeros((niter, Defaults.NCL))
        evt = self.neutrino_sample.event_list
        elo, ehi = Defaults.map_logE_edge[ebin], Defaults.map_logE_edge[ebin + 1]
        flatevt = cy.utils.Events(concat([i.as_dataframe for i in itertools.chain.from_iterable(evt)]))
        flatevt = flatevt[(flatevt['log10energy'] >= elo) * (flatevt['log10energy'] < ehi)]
        galaxy_sample = self.gs
        idx_mask = self.idx_mask

        if mp_cpus > 1:
            p = Pool(mp_cpus)
            iterables = ((flatevt, galaxy_sample, idx_mask, ebin) for i in range(niter))
            cl = p.starmap(bootstrap_worker, iterables)
        else:
            cl = np.zeros((niter, Defaults.NCL))
            for i in tqdm(range(niter)):
                cl[i] = bootstrap_worker(flatevt, galaxy_sample, idx_mask, ebin)
        cl = np.array(cl)

        return np.std(cl, axis=0), np.cov(cl.T)

    def loadSTDInterpolation(self):
        self.std_interps = {}
        self.std_interps_n = {}
        for ebin in range(Defaults.NEbin):
            with open('/home/dguevel/git/nuXgal/syntheticData/w_std_unWISE_z=0.4_v4_ebin{}.json'.format(ebin)) as fp:
                data = json.load(fp)

            self.std_interps[ebin] = interp1d(data['f_inj'], np.array([data['cl_std'][str(n)] for n in data['n_inj']]).T, fill_value='extrapolate', bounds_error=False)
            self.std_interps_n[ebin] = interp1d(data['n_inj'], np.array([data['cl_std'][str(n)] for n in data['n_inj']]).T, fill_value='extrapolate', bounds_error=False)

    def calculate_w_mean(self):
        """Compute the mean cross corrleations assuming neutrino sources follow the same alm
            Note that this is slightly different from the original Cl as the mask has been updated.
        """

        overdensity_g = hp.alm2map(self.gs.overdensityalm, nside=Defaults.NSIDE, verbose=False)
        overdensity_g[self.idx_mask] = hp.UNSEEN
        w_mean = hp.anafast(overdensity_g) / self.f_sky
        self.w_model_f1 = np.zeros((Defaults.NEbin, Defaults.NCL))
        for i in range(Defaults.NEbin):
            self.w_model_f1[i] = w_mean
        return

    def inputData(self, ns, bootstrap_niter=100, mp_cpus=1):
        """Input data

        Parameters
        ----------
        ns : `NeutrinoSample`
            A NeutrinoSample Object

        Returns
        -------
        None
        """

        self.neutrino_sample = ns
        ns.updateMask(self.idx_mask)
        self.w_data = ns.getCrossCorrelation(self.gs)
        self.Ncount = ns.getEventCounts()

        self.w_std = np.copy(self.w_atm_std)
        self.w_std_square = np.copy(self.w_atm_std_square)
        self.w_cov = np.zeros((Defaults.NEbin, Defaults.NCL, Defaults.NCL))

        for ebin in range(self.Ebinmin, self.Ebinmax):
            self.w_std[ebin], self.w_cov[ebin] = self.bootstrapSigma(ebin, niter=bootstrap_niter, mp_cpus=mp_cpus)
            self.w_std_square[ebin] = self.w_std[ebin]**2

    def log_likelihood_Ebin(self, f, energyBin):
        """Compute the log of the likelihood for a particular model in given energy bin

        Parameters
        ----------
        f : `float`
            The fraction of neutrino events correlated with the Galaxy sample
        energyBin: `index`
            The energy bin where likelihood is computed
        Returns
        -------
        logL : `float`
            The log likelihood, computed as sum_l (data_l - f * model_mean_l) /  model_std_l
        """

        f = np.array(f)

        w_data = self.w_data[energyBin, self.lmin:]

        w_model_mean = (self.w_model_f1[self.lmin:] * f)
        w_model_mean += (self.w_atm_mean[energyBin, self.lmin:] * (1 - f))

        w_std = self.w_std[energyBin, self.lmin:]

        lnL_le = norm.logpdf(
            w_data, loc=w_model_mean, scale=w_std)
        return np.sum(lnL_le)

    def log_likelihood(self, f):
        """Compute the log of the likelihood for a particular model

        Parameters
        ----------
        f : `float`
            The fraction of neutrino events correlated with the Galaxy sample

        Returns
        -------
        logL : `float`
            The log likelihood, computed as sum_l (data_l - f * model_mean_l) /  model_std_l
        """

        f = np.array(f)

        lnL_le = 0
        for i, ebin in enumerate(range(self.Ebinmin, self.Ebinmax)):
            w_data = self.w_data[ebin, self.lmin:]

            w_model_mean = (self.w_model_f1[self.lmin:] * f[i])
            w_model_mean += (self.w_atm_mean[ebin, self.lmin:] * (1 - f[i]))

            w_std = self.w_std[ebin, self.lmin:]

            lnL_le += norm.logpdf(
                w_data, loc=w_model_mean, scale=w_std)
        return np.sum(lnL_le)

    def chi_square_Ebin(self, f, energyBin):
        w_model_mean = (self.w_model_f1[energyBin].T * f)
        w_model_mean += (self.w_atm_mean[energyBin].T * (1 - f))
        w_model_std_square = self.w_std_square[energyBin]

        chisquare = (self.w_data[energyBin] - w_model_mean) ** 2 / w_model_std_square
        return np.sum(chisquare[self.lmin:])

    def minimize__lnL_analytic(self):
        len_f = self.Ebinmax - self.Ebinmin
        f = np.zeros(len_f)
        for i, ebin in enumerate(range(self.Ebinmin, self.Ebinmax)):
            cgg = self.w_model_f1[ebin, self.lmin:]
            cgnu = self.w_data[ebin, self.lmin:]
            cgatm = self.w_atm_mean[ebin, self.lmin:]
            cstd = self.w_std[ebin, self.lmin:]
            f[i] = np.sum((cgg-cgatm)*(cgnu-cgatm)/cstd**2)/np.sum((cgg-cgatm)**2/cstd**2)
            #sigma_fhat = np.sqrt(1/np.sum(((cgg-cgatm)**2)/2/cstd**2))

        ts = 2*(self.log_likelihood(f) - self.log_likelihood(np.zeros(len_f)))
        return f, ts

    def minimize__lnL(self):
        """Minimize the log-likelihood
        Parameters
        ----------
        f : `float`
            The fraction of neutrino events correlated with the Galaxy sample
        Returns
        -------
        x : `array`
            The parameters that minimize the log-likelihood
        TS : `float`
            The Test Statistic, computed as 2 * logL_x - logL_0
        """
        len_f = self.Ebinmax - self.Ebinmin
        nll = lambda *args: -self.log_likelihood(*args)
        initial = 0.1 + 0.1 * np.random.randn(len_f)
        soln = minimize(nll, initial, bounds=[(0, 1)] * (len_f))
        #soln = minimize(nll, initial)

        return soln.x, (self.log_likelihood(soln.x) -\
                            self.log_likelihood(np.zeros(len_f))) * 2

    def minimize__lnL_cov(self):
        """Minimize the log-likelihood
        Parameters
        ----------
        f : `float`
            The fraction of neutrino events correlated with the Galaxy sample
        Returns
        -------
        x : `array`
            The parameters that minimize the log-likelihood
        TS : `float`
            The Test Statistic, computed as 2 * logL_x - logL_0
        """
        len_f = self.Ebinmax - self.Ebinmin
        nll = lambda *args: -self.log_likelihood_cov(*args)
        initial = 0.1 + 0.1 * np.random.randn(len_f)
        soln = minimize(nll, initial, bounds=[(0, 1)] * (len_f))
        #soln = minimize(nll, initial)

        return soln.x, (self.log_likelihood_cov(soln.x) -\
                            self.log_likelihood_cov(np.zeros(len_f))) * 2

    def minimize__lnL_free_index(self):
        """Minimize the log-likelihood

        Parameters
        ----------
        f : `float`
            The fraction of neutrino events correlated with the Galaxy sample

        Returns
        -------
        x : `array`
            The parameters that minimize the log-likelihood
        TS : `float`
            The Test Statistic, computed as 2 * logL_x - logL_0
        """
        len_f = (self.Ebinmax - self.Ebinmin)
        nll = lambda *args: -self.log_likelihood(*args)
        initial = 0.5 + 0.1 * np.random.randn(len_f)
        initial = np.hstack([initial, [2.5]])
        bounds = len_f * [[-4, 4],] + [[Defaults.GAMMAS.min(), Defaults.GAMMAS.max()]]
        soln = minimize(nll, initial, bounds=bounds)
        null_x = len_f * [0] + [2.5]
        return soln.x, (self.log_likelihood(soln.x) -\
                            self.log_likelihood(null_x)) * 2

    def plotCastro(self, TS_threshold=4, coloralphalimit=0.01, colorfbin=500):
        """Make a 'Castro' plot of the likelihood

        Parameters
        ----------
        TS_threshold : `float`
            Theshold at which to cut off the colormap
        coloralphalimit : `float`
        colorfbin : `int`
        """
        plt.figure(figsize=(8, 6))
        font = {'family': 'Arial', 'weight' : 'normal', 'size'   : 21}
        legendfont = {'fontsize' : 21, 'frameon' : False}
        matplotlib.rc('font', **font)
        matplotlib.rc('legend', **legendfont)
        matplotlib.rc("text", usetex=True)

        plt.ylabel(r'$E^2 dN/dE\,[\mathrm{GeV\,cm^{-2}\,s^{-1}\,sr^{-1}}]$')
        plt.xlabel(r'$\log$ (E [GeV])')
        #plt.ylim(1e-3, 10) # for f_astro
        plt.ylim(1e-9, 1e-5) # for flux
        plt.xlim(2.5, 5.5)
        plt.yscale('log')

        #cmap = matplotlib.colors.LinearSegmentedColormap.from_list("",
        #["dimgrey", "olive", "forestgreen","yellowgreen"])
        #["white",  "dimgray",  "mediumslateblue",  "cyan", "yellow", "red"]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["navy", "deepskyblue", "lightgrey"])


        bestfit_f, _ = self.minimize__lnL()

        # common x for castro object initialization
        f_Ebin = np.linspace(0, 4, 1000)

        exposuremap = ICECUBE_EXPOSURE_LIBRARY.get_exposure('IC86-2012', 2.28)

        for idx_E in range(self.Ebinmin, self.Ebinmax):
            # exposuremap assuming alpha = 2.28 (numu) to convert bestfit f_astro to flux
            exposuremap_E = exposuremap[idx_E].copy()
            exposuremap_E[self.idx_mask] = hp.UNSEEN
            exposuremap_E = hp.ma(exposuremap_E)
            factor_f2flux = self.Ncount[idx_E] / (exposuremap_E.mean() * 1e4 * Defaults.DT_SECONDS *
                                                  self.N_yr * 4 * np.pi * self.f_sky * Defaults.map_dlogE *
                                                  np.log(10.)) * Defaults.map_E_center[idx_E]

            idx_bestfit_f = idx_E - self.Ebinmin
            lnl_max = self.log_likelihood_Ebin(bestfit_f[idx_bestfit_f], idx_E)
            lnL_Ebin = np.zeros_like(f_Ebin)
            for idx_f, f in enumerate(f_Ebin):
                lnL_Ebin[idx_f] = self.log_likelihood_Ebin(f, idx_E)

            castro = LnLFn(f_Ebin, -lnL_Ebin)
            TS_Ebin = castro.TS()
            # if this bin is significant, plot the 1 sigma interval
            if TS_Ebin > TS_threshold:
                f_lo, f_hi = castro.getInterval(0.32)
                plt.errorbar(Defaults.map_logE_center[idx_E], bestfit_f[idx_bestfit_f] * factor_f2flux,
                             yerr=[[(bestfit_f[idx_bestfit_f]-f_lo) * factor_f2flux],
                                   [(f_hi-bestfit_f[idx_bestfit_f]) * factor_f2flux]],
                             xerr=Defaults.map_dlogE/2., color='k')
                f_select_lo, f_select_hi = castro.getInterval(coloralphalimit)

            # else plot the 2 sigma upper limit
            else:
                f_hi = castro.getLimit(0.05)
                #print (f_hi)
                plt.errorbar(Defaults.map_logE_center[idx_E], f_hi * factor_f2flux, yerr=f_hi * factor_f2flux * 0.2,
                             xerr=Defaults.map_dlogE/2., uplims=True, color='k')
                f_select_lo, f_select_hi = 0, castro.getLimit(coloralphalimit)


            # compute color blocks of delta likelihood
            dlnl = np.zeros((colorfbin, 1))
            f_select = np.linspace(f_select_lo, f_select_hi, colorfbin+1)

            for idx_f_select, _f_select in enumerate(f_select[:-1]):
                dlnl[idx_f_select][0] = self.log_likelihood_Ebin(_f_select, idx_E) - lnl_max

            y_select = f_select * factor_f2flux
            m = plt.pcolormesh([Defaults.map_logE_edge[idx_E], Defaults.map_logE_edge[idx_E+1]], y_select, dlnl,
                               cmap=cmap, vmin=-2.5, vmax=0., linewidths=0, edgecolors='none')

        cbar = plt.colorbar(m)
        cbar.ax.set_ylabel(r'$\Delta\log\,L$', rotation=90, fontsize=16, labelpad=15)
        plt.subplots_adjust(left=0.14, bottom=0.14)
        plt.savefig(os.path.join(Defaults.NUXGAL_PLOT_DIR, 'Fig_sedlnl.png'))

    def log_prior(self, f):
        """Compute log of the prior on a f, implemented as a flat prior between 0 and 1.5

        Parameters
        ----------
        f : `float`
            The signal fraction

        Returns
        -------
        value : `float`
            The log of the prior
        """
        if np.min(f) > -4. and np.max(f) < 4.:
            return 0.
        return -np.inf

    def log_probability(self, f):
        """Compute log of the probablity of f, given some data

        Parameters
        ----------
        f : `float`
            The signal fraction

        Returns
        -------
        value : `float`
            The log of the probability, defined as log_prior + log_likelihood
        """
        lp = self.log_prior(f)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(f)

    def runMCMC(self, Nwalker, Nstep):
        """Run a Markov Chain Monte Carlo

        Parameters
        ----------
        Nwalker : `int`
        Nstep : `int`
        """

        ndim = self.Ebinmax - self.Ebinmin
        pos = 0.3 + np.random.randn(Nwalker, ndim) * 0.1
        nwalkers, ndim = pos.shape
        backend = emcee.backends.HDFBackend(Defaults.CORNER_PLOT_FORMAT.format(galaxyName=self.gs.galaxyName,
                                                                               nyear=str(self.N_yr)))
        backend.reset(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability, backend=backend)
        sampler.run_mcmc(pos, Nstep, progress=True)

    def plotMCMCchain(self, ndim, labels, truths, plotChain=False):
        """Plot the results of a Markov Chain Monte Carlo

        Parameters
        ----------
        ndim : `int`
            The number of variables
        labels : `array`
            Labels for the variables
        truths : `array`
            The MC truth values
        """

        reader = emcee.backends.HDFBackend(Defaults.CORNER_PLOT_FORMAT.format(galaxyName=self.gs.galaxyName,
                                                                              nyear=str(self.N_yr)))
        if plotChain:
            fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
            samples = reader.get_chain()

            for i in range(ndim):
                ax = axes[i]
                ax.plot(samples[:, :, i], "k", alpha=0.3)
                ax.set_xlim(0, len(samples))
                ax.set_ylabel(labels[i])
                ax.yaxis.set_label_coords(-0.1, 0.5)

            axes[-1].set_xlabel("step number")
            fig.savefig(os.path.join(Defaults.NUXGAL_PLOT_DIR, 'MCMCchain.pdf'))

        flat_samples = reader.get_chain(discard=100, thin=15, flat=True)
        #print(flat_samples.shape)
        fig = corner.corner(flat_samples, labels=labels, truths=truths)
        fig.savefig(os.path.join(Defaults.NUXGAL_PLOT_DIR, 'Fig_MCMCcorner.pdf'))

def bootstrap_worker(flatevt, galaxy_sample, idx_mask, ebin):

    ns2 = NeutrinoSample()
    idx = np.random.choice(len(flatevt), size=len(flatevt))
    newevt = flatevt[idx]

    ns2.inputTrial([[newevt]])
    ns2.updateMask(idx_mask)
    # suppress invalid value warning which we get because of
    #  the energy bin filter
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cl = ns2.getCrossCorrelationEbin(galaxy_sample, ebin)
    return cl
