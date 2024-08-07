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
from scipy.stats import norm, distributions, multivariate_normal
from scipy.interpolate import interp1d

from . import Defaults
from .NeutrinoSample import NeutrinoSample
from .FermipyCastro import LnLFn
from .GalaxySample import GALAXY_LIBRARY
from .Exposure import ICECUBE_EXPOSURE_LIBRARY
from .CskyEventGenerator import CskyEventGenerator
from .Models import TemplateModel, MCBackgroundModel
from .DataSpec import data_spec_factory


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
    WCovFname = Defaults.SYNTHETIC_W_COV_FORMAT
    neutrino_sample_class = NeutrinoSample

    def __init__(self, N_yr, galaxyName, Ebinmin, Ebinmax, lmin, gamma=2.5, recompute_model=False, mc_background=False, fit_bounds=[0, 1], path_sig=''):
        """
        Initialize the Likelihood object.

        Parameters
        ----------
        N_yr : float
            Number of years to simulate if computing the models.
        galaxyName : str
            Name of the Galaxy sample.
        Ebinmin : int
            Index of the minimum energy bin for likelihood computation.
        Ebinmax : int
            Index of the maximum energy bin for likelihood computation.
        lmin : int
            Minimum value of l to be taken into account in likelihood.
        gamma : float, optional
            Spectral index of the neutrino flux. Default is 2.5.
        recompute_model : bool, optional
            If True, recompute the model. Best done on a cluster. Default is False.
        mc_background : bool, optional
            If True, use Monte Carlo background model. Default is False.
        fit_bounds : list, optional
            List of fit bounds for each energy bin. Default is [0, 1].
        """
        self.N_yr = N_yr
        self.gs = GALAXY_LIBRARY.get_sample(galaxyName)
        self.anafastMask()
        self.Ebinmin = Ebinmin
        self.Ebinmax = Ebinmax
        self.lmin = lmin
        self._event_generator = None
        self._per_ebin_event_generators = None
        self.w_data = None
        self.Ncount = None
        self.gamma = gamma
        self.mc_background = mc_background
        self.acceptance = None
        self.path_sig = path_sig
        if fit_bounds is not None:
            self.fit_bounds = [fit_bounds] * (Ebinmax - Ebinmin)
        else:
            self.fit_bounds = None

        self.background_model = MCBackgroundModel(
            self.gs,
            self.N_yr,
            self.idx_mask,
            recompute=recompute_model,
            path_sig=path_sig
        )

        self.w_atm_mean = self.background_model.w_mean
        self.w_atm_std = self.background_model.w_std
        self.w_atm_std_square = self.w_atm_std ** 2

        self.signal_model = TemplateModel(
            self.gs,
            self.N_yr,
            self.idx_mask,
            recompute=recompute_model,
            path_sig=path_sig,
            gamma=gamma
        )

        self.w_model_f1 = self.signal_model.w_mean
        self.w_model_f1_std = self.signal_model.w_std

        self.cov_fname = self.WCovFname.format(nyear=self.N_yr, galaxyName=galaxyName)
        if os.path.exists(self.cov_fname):
            self.w_cov = np.load(self.cov_fname)
        else:
            self.w_cov = np.zeros((Defaults.NEbin, Defaults.NCL, Defaults.NCL))


    @property
    def event_generator(self):
        if self._event_generator is None:
            self._event_generator = CskyEventGenerator(
                self.N_yr,
                self.gs,
                gamma=self.gamma,
                Ebinmin=self.Ebinmin,
                Ebinmax=self.Ebinmax,
                idx_mask=self.idx_mask,
                mc_background=self.mc_background,
                path_sig=self.path_sig)
        return self._event_generator
    
    @property
    def per_ebin_event_generators(self):
        if self._per_ebin_event_generators is None:
            self._per_ebin_event_generators = []
            for i in range(Defaults.NEbin):
                eg = CskyEventGenerator(
                    self.N_yr,
                    self.gs,
                    gamma=self.gamma,
                    Ebinmin=i,
                    Ebinmax=i+1,
                    idx_mask=self.idx_mask,
                    mc_background=self.mc_background,
                    path_sig=self.path_sig)

                self._per_ebin_event_generators.append(eg)
        return self._per_ebin_event_generators

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

        if kwargs['fit_bounds']:
            fit_bounds = [0, 1]
        else:
            fit_bounds = None

        llh = Likelihood(
            kwargs['N_yr'],
            kwargs['galaxy_catalog'],
            kwargs['ebinmin'],
            kwargs['ebinmax'],
            kwargs['lmin'],
            gamma=kwargs['gamma'],
            fit_bounds=fit_bounds,
            mc_background=kwargs['mc_background'],
            path_sig=kwargs['path_sig'])

        llh.w_data = np.zeros((Defaults.NEbin, Defaults.NCL))
        llh.w_std = np.zeros((Defaults.NEbin, Defaults.NCL))
        llh.w_cov = np.zeros((Defaults.NEbin, Defaults.NCL, Defaults.NCL))
        for i, ebin in enumerate(range(llh.Ebinmin, llh.Ebinmax)):
            if isinstance(list(kwargs['cls'].keys())[i], str):
                ebin = str(ebin)
            llh.w_data[int(ebin)] = kwargs['cls'][ebin]
            if 'std' in kwargs:
                llh.w_std[int(ebin)] = kwargs['cls_std'][ebin]
            if 'cov' in kwargs:
                llh.w_cov[int(ebin)] = kwargs['cov'][ebin]
            else:
                cov_fname = llh.WCovFname.format(nyear=kwargs['N_yr'], galaxyName=kwargs['galaxy_catalog'])
                if os.path.exists(cov_fname):
                    llh.w_cov = np.load(cov_fname)
                else:
                    llh.w_cov = np.ones((Defaults.NEbin, Defaults.NCL, Defaults.NCL))

        if 'countsmap' in kwargs:
            ns = NeutrinoSample()
            countsmap = np.array(kwargs['countsmap'])
            countsmap[countsmap == None] = hp.UNSEEN
            countsmap = hp.ma(countsmap)
            ns.inputCountsmap(countsmap)
            llh.inputData(ns, bootstrap_niter=0)

        llh.neutrino_sample = llh.neutrino_sample_class()
        llh.neutrino_sample.build_aeff_matrix(llh.event_generator.ana)
        llh.neutrino_sample.build_aeff_map()

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
            iterables = ((flatevt, galaxy_sample, idx_mask, ebin, self.acceptance) for i in range(niter))
            cl = p.starmap(bootstrap_worker, iterables)
        else:
            cl = np.zeros((niter, Defaults.NCL))
            for i in tqdm(range(niter)):
                cl[i] = bootstrap_worker(flatevt, galaxy_sample, idx_mask, ebin, self.acceptance)
        cl = np.array(cl)

        return np.std(cl, axis=0), np.cov(cl.T)

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
        self.w_data = ns.getCrossCorrelation(self.gs, acceptance=self.acceptance)
        self.Ncount = ns.getEventCounts()

        self.w_std = np.copy(self.w_atm_std)
        self.w_std_square = np.copy(self.w_atm_std_square)
        self.w_cov = np.zeros((Defaults.NEbin, Defaults.NCL, Defaults.NCL))

        for ebin in range(self.Ebinmin, self.Ebinmax):
            if bootstrap_niter > 0:
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

        w_model_mean = (self.w_model_f1[energyBin, self.lmin:] * f)
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

            w_model_mean = (self.w_model_f1[ebin, self.lmin:] * f[i])
            w_model_mean += (self.w_atm_mean[ebin, self.lmin:] * (1 - f[i]))

            w_std = self.w_std[ebin, self.lmin:]

            lnL_le += norm.logpdf(
                w_data, loc=w_model_mean, scale=w_std)
        return np.sum(lnL_le)

    def log_likelihood_free_atm(self, fcorr, fatm):
        lnL_le = 0
        for i, ebin in enumerate(range(self.Ebinmin, self.Ebinmax)):
            lnL_le += self.log_likelihood_free_atm_Ebin(fcorr[i], fatm[i], ebin)
        return lnL_le

    def log_likelihood_free_atm_Ebin(self, fcorr, fatm, energyBin):

        w_data = self.w_data[energyBin, self.lmin:]

        w_model_mean = (self.w_model_f1[energyBin, self.lmin:] * fcorr)
        w_model_mean += (self.w_atm_mean[energyBin, self.lmin:] * fatm)

        lnL_le = self.multi_norm[energyBin - self.Ebinmin].logpdf(w_data - w_model_mean)

        return lnL_le

    
    def minimize__lnL_free_atm(self):
        self.multi_norm = [multivariate_normal(cov=self.w_cov[i, self.lmin:, self.lmin:], allow_singular=True) for i in range(self.Ebinmin, self.Ebinmax)]

        def minfunc(pars):
            return -self.log_likelihood_free_atm(pars[0:3], pars[3:6])
        initial = [0 for i in range(Defaults.NEbin)]
        initial += [1 for i in range(Defaults.NEbin)]

        res = minimize(minfunc, initial)

        initial = [1 for i in range(Defaults.NEbin)]

        def minfunc2(pars):
            return -self.log_likelihood_free_atm([0, 0, 0], pars[0:3])

        res_bg = minimize(minfunc2, initial)

        ts = -2 * (res.fun - res_bg.fun)
        return res.x, ts

    def log_likelihood_cov(self, f):
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

            w_model_mean = (self.w_model_f1[ebin, self.lmin:] * f[i])
            w_model_mean += (self.w_atm_mean[ebin, self.lmin:] * (1 - f[i]))

            w_cov = self.w_cov[ebin, self.lmin:, self.lmin:]

            #lnL_le += multivariate_normal.logpdf(
            #    w_data, mean=w_model_mean, cov=w_cov, allow_singular=True)
            lnL_le += self.multi_norm[i].logpdf(w_data - w_model_mean)
        return lnL_le


    def log_likelihood_cov_Ebin(self, f, energyBin):
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

        w_data = self.w_data[energyBin, self.lmin:]

        w_model_mean = (self.w_model_f1[energyBin, self.lmin:] * f)
        w_model_mean += (self.w_atm_mean[energyBin, self.lmin:] * (1 - f))

        lnL_le = self.multi_norm[energyBin].logpdf(w_data - w_model_mean)
        return lnL_le

    def log_likelihood_free_bg_ns_gamma(self, pars):
        fcorr, gamma = pars[:2]
        fcorr = self.fi_given_f_gamma(fcorr, gamma)
        fatm, gammaatm = pars[2:]
        fatm = self.fi_given_f_gamma(fatm, gammaatm)

        lnL_le = 0
        for i, ebin in enumerate(range(self.Ebinmin, self.Ebinmax)):
            w_data = self.w_data[ebin, self.lmin:]

            w_model_mean = self.w_model_f1[ebin, self.lmin:] * fcorr[i]
            w_model_mean += self.w_atm_mean[ebin, self.lmin:] * fatm[i]

            w_cov = self.w_cov[ebin, self.lmin:, self.lmin:]

            #lnL_le += multivariate_normal.logpdf(
            #    w_data - w_model_mean, cov=w_cov, allow_singular=False)
            lnL_le += self.multi_norm[i].logpdf(w_data - w_model_mean)
        return lnL_le

    def minimize__lnL_free_bg_ns_gamma(self):
        self.multi_norm = [multivariate_normal(cov=self.w_cov[i, self.lmin:, self.lmin:], allow_singular=True) for i in range(self.Ebinmin, self.Ebinmax)]
        res = minimize(lambda pars: -self.log_likelihood_free_bg_ns_gamma(pars), [.1, 2.5, 1, 3.0], bounds=((0, None), (1, 4), (0, None), (1, 4)), method='Nelder-Mead')
        res2 = minimize(lambda pars: -self.log_likelihood_free_bg_ns_gamma([0, 2.5] + list(pars)), [1, 3.0], bounds=((0, None), (1, 4)), method='Nelder-Mead')
        ts = -2 * (res.fun - res2.fun)
        return res.x, ts

    def chi_square_Ebin(self, f, energyBin):
        """
        Calculate the chi-square value for a given energy bin.

        Parameters:
        ----------
        f : float
            The fraction of the model to use.
        energyBin : int
            The index of the energy bin.

        Returns:
        -------
        float
            The chi-square value.
        """

        w_data = self.w_data[energyBin, self.lmin:]

        w_model_mean = (self.w_model_f1[energyBin, self.lmin:] * f)
        w_model_mean += (self.w_atm_mean[energyBin, self.lmin:] * (1 - f))

        w_std = self.w_std[energyBin, self.lmin:]

        chisquare = (w_data - w_model_mean) ** 2 / w_std ** 2
        return np.sum(chisquare)
    
    def chi_square_cov_Ebin(self, f, energyBin):
        """
        Calculate the chi-square value for a given energy bin including
        a covariance matrix.

        Parameters
        ----------
        f : float
            The fraction of the model to be used.
        energyBin : int
            The index of the energy bin.

        Returns
        -------
        chi_square : float
            The calculated chi-square value.
        """
        w_data = self.w_data[energyBin, self.lmin:]

        w_model_mean = (self.w_model_f1[energyBin, self.lmin:] * f)
        w_model_mean += (self.w_atm_mean[energyBin, self.lmin:] * (1 - f))

        w_cov = self.w_cov[energyBin, self.lmin:, self.lmin:]
        z = w_data - w_model_mean
        chi_square = np.matmul(z, np.linalg.solve(w_cov, z))

        return chi_square

    def chi_square_ns_gamma(self, ns, gamma):
        """
        Calculate the chi-square value for a given number of neutrino events
        and spectral index.

        Parameters
        ----------
        ns : float
            The number of neutrino events.
        gamma : float
            The spectral index of the neutrino flux.

        Returns
        -------
        chi_square : float
            The calculated chi-square value.
        """

        f = self.f_given_ns_gamma(ns, gamma)
        chi_square = 0
        for i in range(self.Ebinmin, self.Ebinmax):
            chi_square += self.chi_square_cov_Ebin(f[i], i)
        return chi_square

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
        soln = minimize(nll, initial, bounds=self.fit_bounds)

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
        #w_cov = self.w_cov[:, self.lmin:, self.lmin:].copy()
        #for i in range(self.Ebinmin, self.Ebinmax):
        #    w_cov[i] += np.eye(w_cov[i].shape[0]) * 1e-9
        #self.multi_norm = [multivariate_normal(cov=w_cov[i], allow_singular=True) for i in range(self.Ebinmin, self.Ebinmax)]
        self.multi_norm = [multivariate_normal(cov=self.w_cov[i, self.lmin:, self.lmin:], allow_singular=True) for i in range(self.Ebinmin, self.Ebinmax)]
        soln = minimize(nll, initial, bounds=self.fit_bounds)

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


    def mle_ns_given_gamma(self, gamma):
        a = self.w_data[:, self.lmin:]
        b = self.w_model_f1[:, self.lmin:]
        c = self.w_atm_mean[:, self.lmin:]
        w_cov_inv = np.linalg.inv(self.w_cov[:, self.lmin:, self.lmin:])
        acc_total = self.acc_total(gamma)
        acc_ebin = np.array([self.acc_ebin(i, gamma) for i in range(self.Ebinmin, self.Ebinmax)])
        k = acc_ebin / acc_total
        N = self.Ncount.sum()
        numerator = 0
        denominator = 0
        for i in range(self.Ebinmin, self.Ebinmax):
            numerator += -k[i] * np.matmul(b[i] + c[i], np.matmul(w_cov_inv[i], (a[i] - b[i])))
            denominator += k[i]**2 * np.matmul(b[i] + c[i], np.matmul(w_cov_inv[i], (b[i] + c[i])))

        # restrict the domain of the solution to be positive
        ns = max(0, N * numerator / denominator)
        return ns


    def minimize__lnL_ns_gamma(self, method='Nelder-Mead'):
        """Minimize the log-likelihood

        Parameters
        ----------
        method : `str`
            The optimization method to use; Nelder-Mead seems to work best

        Returns
        -------
        x : `array`
            The parameters that minimize the log-likelihood
        TS : `float`
            The Test Statistic, computed as 2 * logL_x - logL_0
        """

        def minfunc(params):
            ns, gamma = params
            return -self.log_likelihood_ns_gamma(ns, gamma)

        initial = [2000, 2.5]
        bounds = [[0, None], [1, 4]]
        soln = minimize(minfunc, initial, bounds=bounds, method=method)
        return soln.x, (self.log_likelihood_ns_gamma(*soln.x) -\
                            self.log_likelihood_ns_gamma(0, 2.5)) * 2

    def log_likelihood_ns_gamma(self, ns, gamma):
        """Compute the log of the likelihood for a particular model

        Parameters
        ----------
        ns : `float`
            The number of neutrino events
        gamma : `float`
            The spectral index of the neutrino flux

        Returns
        -------
        logL : `float`
            The log likelihood, computed as sum_l (data_l - f * model_mean_l) /  model_std_l
        """
        f = self.f_given_ns_gamma(ns, gamma)
        return self.log_likelihood_cov(f)

    def log_likelihood_free_bg(self, pars):
        fcorr, gamma = pars[:2]
        fcorr = self.f_given_ns_gamma(fcorr, gamma)
        fatm = pars[2:]

        lnL_le = 0
        for i, ebin in enumerate(range(self.Ebinmin, self.Ebinmax)):
            w_data = self.w_data[ebin, self.lmin:]

            w_model_mean = self.w_model_f1[ebin, self.lmin:] * fcorr[i]
            w_model_mean += self.w_atm_mean[ebin, self.lmin:] * fatm[i]

            lnL_le += self.multi_norm[i].logpdf(w_data - w_model_mean)
        return lnL_le

    def minimize__lnL_free_bg(self):
        self.multi_norm = [multivariate_normal(cov=self.w_cov[i, self.lmin:, self.lmin:], allow_singular=True) for i in range(self.Ebinmin, self.Ebinmax)]

        res = minimize(lambda pars: -self.log_likelihood_free_bg(pars), [1000, 2.5, 1, 1, 1], bounds=((0, None), (1, 4), (None, None), (None, None), (None, None)), method='Nelder-Mead')
        res2 = minimize(lambda pars: -self.log_likelihood_free_bg([0, 2.5] + list(pars)), [1, 1, 1], bounds=((None, None), (None, None), (None, None)), method='Nelder-Mead')
        ts = -2 * (res.fun - res2.fun)
        return res.x, ts

    def f_given_ns_gamma(self, ns, gamma):
        """
        Convert a given astrophysical neutrino count to a signal fraction in 
        each energy bin with spectral index gamma
        
        Parameters
        ----------
        ns : `float`
            The number of neutrino events
        gamma : `float`
            The spectral index of the neutrino flux

        Returns
        -------
        f : `array`
            The signal fraction in each energy bin
        """
        n_total = self.Ncount
        acc_total = self.acc_total(gamma)
        acc_ebin = np.array([self.acc_ebin(i, gamma) for i in range(self.Ebinmin, self.Ebinmax)])
        f = ns * acc_ebin / acc_total / n_total
        return f
    
    #def f_given_ns_gamma(self, ns, gamma):
        #n_total = self.Ncount
        #if not hasattr(self, '_acc_total_interp'):
        #    self._build_acc_interps()
        #acc_ebin = np.array([self._acc_ebin_interp[i](gamma) for i in range(self.Ebinmin, self.Ebinmax)])
        #acc_total = self._acc_total_interp(gamma)
        #f = ns * acc_ebin / acc_total / n_total
        #return f

        #if not hasattr(self, '_acc_total_interp'):
        #    self._build_acc_interps()
        #acc_ebin_weight = np.array([self._acc_ebin_interp[i](gamma) for i in range(self.Ebinmin, self.Ebinmax)])
        #acc_total_weight = self._acc_total_interp(gamma)
        #f = ns * acc_ebin_weight / acc_total_weight / self.Ncount
        #return f
    
    def fi_given_f_gamma(self, f, gamma):
        """
        Convert a given astrophysical neutrino count to a signal fraction in 
        each energy bin with spectral index gamma
        
        Parameters
        ----------
        ns : `float`
            The number of neutrino events
        gamma : `float`
            The spectral index of the neutrino flux

        Returns
        -------
        f : `array`
            The signal fraction in each energy bin
        """
        acc_total = self.acc_total(gamma)
        acc_ebin = np.array([self.acc_ebin(i, gamma) for i in range(self.Ebinmin, self.Ebinmax)])
        f = f * acc_ebin / acc_total
        return f
    
    def f2n(self, f, gamma):
        factor = self.acc_aeff_weighted(gamma) / np.mean(self.neutrino_sample.countsmap)
        return f / factor
    
    def n2f(self, n, gamma):
        factor = self.acc_aeff_weighted(gamma) / np.mean(self.neutrino_sample.countsmap)
        return n * factor
    
    def fi_given_f_gamma(self, f, gamma):
        acc_total = self.acc_aeff_weighted(gamma)
        acc_ebin = np.array([self.acc_ebin_aeff_weighted(i, gamma) for i in range(self.Ebinmin, self.Ebinmax)])

        if not hasattr(self, '_acc_total_interp'):
            self._build_acc_interps()
        acc_ebin = np.array([self._acc_ebin_interp[i](gamma) for i in range(self.Ebinmin, self.Ebinmax)])
        acc_total = self._acc_total_interp(gamma)
        f = f * acc_ebin / acc_total
        return f

        #fi = []
        #for i in range(self.Ebinmin, self.Ebinmax):
        #    Aeff = self.neutrino_sample.effective_area_map[i]
        #    Aeff[self.idx_mask] = hp.UNSEEN
        #    Aeff = hp.ma(Aeff)
        #    denom = np.mean(self.Ncount[i] / Aeff[i])
        #    numer = np.mean(self.acc_ebin(i, gamma) / Aeff[i])
        #    fi.append(f * numer / denom)

        #Aeff = self.neutrino_sample.effective_area_map.sum(axis=0)
        #Aeff[self.idx_mask] = hp.UNSEEN
        #Aeff = hp.ma(Aeff)
        #f_tot = np.mean(self.acc_total(gamma) / Aeff)
        #return np.array(fi) / f_tot

        #fi = []
        #for i in range(self.Ebinmin, self.Ebinmax):
        #    elo, ehi = 10**Defaults.map_logE_edge[i], 10**Defaults.map_logE_edge[i + 1]
        #    fi.append((ehi**(2-gamma) - elo**(2-gamma)) / (2 - gamma))
        #fi = np.array(fi)

        #flux_tot = self.Ncount / Defaults.NPIXEL

        #return f * fi / flux_tot

        #fi = []
        #for i, ebin in enumerate(range(self.Ebinmin, self.Ebinmax)):
        #    phi0 = self.per_ebin_event_generators[ebin].trial_runner.to_dNdE(f, unit=1, gamma=gamma) * self.event_generator.ana[0].livetime
        #    elo = Defaults.map_logE_edge[ebin]
        #    ehi = Defaults.map_logE_edge[ebin + 1]
        #    E0 = 1
        #    if gamma == 2.:
        #        dNdE = np.log(ehi/elo) * phi0 / E0
        #    else:
        #        dNdE = phi0 / (2-gamma) / E0**(1-gamma) * (ehi**(2-gamma) - elo**(2-gamma))
        #    fi.append(dNdE / self.Ncount[i])
        #return fi
        

    def _build_acc_interps(self):
        from scipy.interpolate import interp1d
        gammas = np.arange(1, 4.25, 0.25)
        self._acc_ebin_interp = []
        for i in range(Defaults.NEbin):
            accs = []
            for g in gammas:
                accs.append(self.acc_ebin_aeff_weighted(i, g))
            self._acc_ebin_interp.append(interp1d(gammas, accs, kind='linear'))

        accs = []
        for g in gammas:
            accs.append(self.acc_aeff_weighted(g))
        self._acc_total_interp = interp1d(gammas, accs)

    def acc_ebin_aeff_weighted(self, ebin, gamma):
        ra, dec = hp.pix2ang(Defaults.NSIDE, np.arange(Defaults.NPIXEL), lonlat=True)
        evt = cy.utils.Events(sindec=np.sin(np.radians(dec)))
        acc_map = np.zeros(Defaults.NPIXEL)
        for subana in self.per_ebin_event_generators[ebin].ana:
            acc_map += subana.acc_param(evt, gamma=gamma)
        acc_map[self.idx_mask] = hp.UNSEEN
        acc_map = hp.ma(acc_map)

        nu_map = self.event_generator.density_nu
        weighted_acc = np.mean(nu_map / acc_map)
        return weighted_acc
    
    def acc_aeff_weighted(self, gamma):
        ra, dec = hp.pix2ang(Defaults.NSIDE, np.arange(Defaults.NPIXEL), lonlat=True)
        evt = cy.utils.Events(sindec=np.sin(np.radians(dec)))
        acc_map = np.zeros(Defaults.NPIXEL)
        for subana in self.event_generator.ana:
            acc_map += subana.acc_param(evt, gamma=gamma)

        acc_map[self.idx_mask] = hp.UNSEEN
        acc_map = hp.ma(acc_map)

        nu_map = self.event_generator.density_nu
        weighted_acc = np.mean(nu_map / acc_map)
        return weighted_acc



    def acc_ebin(self, ebin, gamma):
        return self.per_ebin_event_generators[ebin].trial_runner.get_acc_total(gamma=gamma)

    def acc_total(self, gamma):
        return np.sum([self.acc_ebin(i, gamma) for i in range(self.Ebinmin, self.Ebinmax)])

    def _get_acc(self):
        self._acceptances = []
        self._n_total = []

        for i in range(Defaults.NEbin):
            data_specs = data_spec_factory(i, i + 1)

            self.dataspec = {
                3: data_specs.ps_3yr,
                10: data_specs.ps_10yr,
                'v4': data_specs.ps_v4,
                'ps_v4': data_specs.ps_v4,
                'estes_10yr': data_specs.estes_10yr,
                'dnn_cascade_10yr': data_specs.dnn_cascade_10yr,
                'nt_v5': data_specs.nt_v5}[self.N_yr]

            version = {
                'v4': 'version-004-p02',
                'ps_v4': 'version-004-p02',
                'estes_10yr': 'version-001-p03',
                'dnn_cascade_10yr': 'version-001-p01',
                'nt_v5': 'version-005-p01'}[self.N_yr]
            anas = cy.get_analysis(cy.selections.repo, version, self.dataspec, analysis_region_template=~self.density_nu.mask)

            self._n_total.append(np.sum([len(ana.data) for ana in anas]))
            self._acceptances.append([])
            for ana in anas:
                evts = ana.sig
                self._acceptances[-1].append(cy.pdf.SinDecAccParametrization(evts))

    def chi_square_free_bg(self, pars):
        constant = 0
        k = Defaults.NCL - self.lmin
        for i in range(self.Ebinmin, self.Ebinmax):
            constant += -1/2 * np.linalg.slogdet(self.w_cov[i, self.lmin:, self.lmin:])[1]
            constant += -k / 2 * np.log(2 * np.pi)
        chi_square = self.log_likelihood_free_bg(pars) - constant
        return -2*chi_square

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

def bootstrap_worker(flatevt, galaxy_sample, idx_mask, ebin, acceptance):

    ns2 = NeutrinoSample()
    idx = np.random.choice(len(flatevt), size=len(flatevt))
    newevt = flatevt[idx]

    ns2.inputTrial([[newevt]])
    ns2.updateMask(idx_mask)
    # suppress invalid value warning which we get because of
    #  the energy bin filter
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cl = ns2.getCrossCorrelationEbin(galaxy_sample, ebin, acceptance)
    return cl
