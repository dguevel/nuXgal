"""Likelihood class for tomographic cross correlation"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import trapz
from astropy.cosmology import WMAP9 as cosmo
import astropy.constants as const
import astropy.units as u

from .TomographicEventGenerator import TomographicEventGenerator
from .Likelihood import Likelihood
from .GalaxySample import GALAXY_LIBRARY
from . import Defaults


class TomographicLikelihood():
    def __init__(self, N_yr, galaxyNames, Ebinmin, Ebinmax, lmin, gamma, fit_bounds, mc_background, recompute_model, weights='cutoff_pl', m=3, k=-1):
        self.N_yr = N_yr
        self.galaxyNames = galaxyNames
        self.Ebinmin = Ebinmin
        self.Ebinmax = Ebinmax
        self.lmin = lmin
        self.gamma = gamma
        self.fit_bounds = fit_bounds
        self.mc_background = mc_background
        self.recompute_model = recompute_model
        self._event_generator = None

        self.n_zbins = len(galaxyNames)
        self.z_center, self.z_width = [], []
        for gname in galaxyNames:
            gs = GALAXY_LIBRARY.get_sample(gname)
            self.z_center.append(gs.z)
            self.z_width.append(gs.dz)
        self.z_center = np.array(self.z_center)
        self.z_width = np.array(self.z_width)

        if weights == 'SFR':
            self.relative_weights = self.calc_weights_SFR()
        elif weights == 'flat':
            self.relative_weights = self.calc_weights_flat()
        elif weights == 'cutoff_pl':
            self.m, self.k = m, k
            self.relative_weights = self.calc_weights_cutoff_pl(m=self.m, k=self.k)
        else:
            raise ValueError('Invalid redshift weighting scheme')

        self.llhs = []
        for i in range(len(self.galaxyNames)):
            self.llhs.append(Likelihood(self.N_yr, self.galaxyNames[i], self.Ebinmin, self.Ebinmax, self.lmin, self.gamma, self.recompute_model, self.mc_background, self.fit_bounds))

    def _get_event_generators(self):
        #for i in range(self.n_zbins):
        #    self.llhs[i]._get_event_generators()
        self.event_generators = [self.llhs[i].event_generator for i in range(self.n_zbins)]
        self.event_generator = TomographicEventGenerator(self.event_generators, self.relative_weights)

    def inputData(self, ns, bootstrap_niter=0):

        self.neutrino_sample = ns
        ns.updateMask(self.llhs[0].idx_mask)
        self.Ncount = ns.getEventCounts()

        for i in range(self.n_zbins):
            self.llhs[i].inputData(ns, bootstrap_niter=bootstrap_niter)

        self.w_data = np.zeros((self.n_zbins, self.Ebinmax - self.Ebinmin, Defaults.NCL))
        for i in range(self.n_zbins):
            self.w_data[i] = self.llhs[i].w_data

    def minimize__lnL_ns_gamma(self, method='Nelder-Mead'):
        # create multinorm object by side effect. TODO fix this
        for i in range(self.n_zbins):
            self.llhs[i].fit_bounds = None
            self.llhs[i].minimize__lnL_cov()

        def minfunc(params):
            ns = params[:self.n_zbins]
            gamma = params[self.n_zbins]
            llh = 0
            for i in range(self.n_zbins):
                llh += self.llhs[i].log_likelihood_ns_gamma(ns[i], gamma)
            return -llh

        initial = 200 * self.relative_weights
        initial = np.append(initial, [2.5])
        bounds = []
        for i in range(self.n_zbins):
            bounds.append([0, None])
        bounds.append([1, 4])
        soln = minimize(minfunc, initial, bounds=bounds, method=method)
        return soln.x, (self.log_likelihood_ns_gamma(soln.x[:-1], soln.x[-1]) -
                        self.log_likelihood_ns_gamma(np.zeros(self.n_zbins), 2.5)) * 2

    def log_likelihood_ns_gamma(self, ns, gamma):
        """
        Calculate the log likelihood for a given number of signal events and
        spectral index assuming independence between energy bins.

        Parameters
        ----------
        ns : array-like
            Number of signal events in each energy bin
            gamma : float
            Spectral index of the signal

            Returns
            -------
        llh : float
            The log likelihood
        """

        llh = 0
        for i in range(self.n_zbins):
            llh += self.llhs[i].log_likelihood_ns_gamma(ns[i], gamma)
        return llh

    def BIC_ns_gamma(self, ns, gamma):
        n_params = self.n_zbins + 1
        n_obs = self.n_zbins * Defaults.NEbin * (Defaults.NCL - self.lmin)
        return n_params * np.log(n_obs) - 2 * self.log_likelihood_ns_gamma(ns, gamma)

    def minimize__lnL_linked(self, method='Nelder-Mead'):
        # create multinorm object by side effect. TODO fix this
        for i in range(self.n_zbins):
            self.llhs[i].fit_bounds = None
            self.llhs[i].minimize__lnL_cov()

        def minfunc(params):
            ns = params[0]
            gamma = params[1]
            return -self.log_likelihood_linked(ns, gamma)

        initial = [2000, 2.5]
        bounds = [[0, None], [1, 4]]
        soln = minimize(minfunc, initial, bounds=bounds, method=method)
        return soln.x, - 2 * (minfunc(soln.x) - minfunc([0, 2.5]))

    def log_likelihood_linked(self, ns, gamma):
        """
        Log likelihood keeping relative redshift bin normalizations 
        fixed to SFR.

        Parameters
        ----------
        ns : float
            Number of astrophysical neutrinos
        gamma : float
            Astrophysical spectral index

        Returns
        -------
        llh : float
            Log likelihood value
        """
        nss = ns * self.calc_weights_SFR()
        llh = 0
        for i in range(self.n_zbins):
            llh += self.llhs[i].log_likelihood_ns_gamma(nss[i], gamma)
        return llh

    def BIC_linked(self, ns, gamma):
        n_obs = self.n_zbins * Defaults.NEbin * (Defaults.NCL - self.lmin)
        return 2 * np.log(n_obs) - 2 * self.log_likelihood_linked(ns, gamma)

    def minimize__lnL_independent(self):
        nss, gammas = [], []
        for i in range(self.n_zbins):
            ns, gamma = self.llhs[i].minimize__lnL_ns_gamma()[0]
            nss.append(ns)
            gammas.append(gamma)

        params = []
        params.extend(nss)
        params.extend(gammas)
        return np.array(params), 2 * (self.log_likelihood_independent(nss, gammas) - self.log_likelihood_independent(np.zeros(self.n_zbins), np.ones(self.n_zbins) * 2.5))

    def BIC_independent(self, nss, gammas):
        n_params = self.n_zbins * 2
        n_obs = self.n_zbins * Defaults.NEbin * (Defaults.NCL - self.lmin)
        return n_params * np.log(n_obs) - 2 * self.log_likelihood_independent(nss, gammas)

    def log_likelihood_independent(self, nss, gammas):
        llh = 0
        for i in range(self.n_zbins):
            llh += self.llhs[i].log_likelihood_ns_gamma(nss[i], gammas[i])
        return llh

    def minimize__lnL_flat(self, method='Nelder-Mead'):
        # create multinorm object by side effect. TODO fix this
        for i in range(self.n_zbins):
            self.llhs[i].fit_bounds = None
            self.llhs[i].minimize__lnL_cov()

        def minfunc(params):
            ns, gamma = params
            return -self.log_likelihood_flat(ns, gamma)

        initial = [2000, 2.5]
        bounds = [[0, None], [1, 4]]
        soln = minimize(minfunc, initial, bounds=bounds, method=method)
        return soln.x, - 2 * (minfunc(soln.x) - minfunc([0, 2.5]))

    def log_likelihood_flat(self, ns, gamma):
        llh = 0
        nss = ns * self.calc_weights_flat()
        for i in range(self.n_zbins):
            llh += self.llhs[i].log_likelihood_ns_gamma(nss[i], gamma)
        return llh

    def BIC_flat(self, ns, gamma):
        n_obs = self.n_zbins * Defaults.NEbin * (Defaults.NCL - self.lmin)
        return 2 * np.log(n_obs) - 2 * self.log_likelihood_flat(ns, gamma)

    @staticmethod
    def init_from_run(**data):

        llh = TomographicLikelihood(
            data['N_yr'],
            data['galaxy_catalog'],
            data['ebinmin'],
            data['ebinmax'],
            data['lmin'],
            data['gamma'],
            data['fit_bounds'],
            data['mc_background'],
            data['recompute_model']
        )

        for i in range(data['n_zbins']):
            llh.llhs[i].w_data = np.zeros((llh.Ebinmax - llh.Ebinmin, Defaults.NCL))
            for j, ebin in enumerate(range(data['ebinmin'], data['ebinmax'])):
                llh.llhs[i].w_data[j] = np.array(data['cls'][str(ebin)][i])

            llh.llhs[i].Ncount = np.array([data['n_total_i'][str(ebin)] for ebin in range(Defaults.NEbin)])

        return llh

    def dphidzdOmega(self, z, Enu, rho0=1e-6/u.Mpc**3, gamma=2.3, E0=100*u.TeV, phi=2e36/u.GeV/u.sr/u.second, a=0.00170, b=0.13, c=3.3, d=5.3):
        """Photon density"""
        return const.c / np.pi / 4 * (1 + z) * self.rho_dot(z, rho0=rho0, a=a, b=b, c=c, d=d) * np.abs(self.dtdz(z)) * self.dNdE(Enu * (1+z), gamma=gamma, E0=E0, phi=phi)

    def dNdE(self, E, gamma=2.3, E0=100*u.TeV, phi=2e36/u.GeV/u.sr/u.second):
        """Power law differential flux"""
        return phi * (E/E0)**(-gamma)

    def g(self, z, a=0.00170, b=0.13, c=3.3, d=5.3):
        """Cole 2001 source evolution function"""
        return (a + b*z) * cosmo.h / (1 + (z/c)**d)

    def rho_dot(self, z, rho0=1e-6/u.Mpc**3, a=0.00170, b=0.13, c=3.3, d=5.3):
        """Star formation rate density"""
        return rho0 * self.g(z, a, b, c, d)

    def dtdz(self, z):
        """Time derivative of redshift"""
        return 1 / (cosmo.H0 * (1+z) * np.sqrt(cosmo.Om0*(1+z)**3 + cosmo.Ode0))

    def calc_weights_SFR(self, a=0.00170, b=0.13, c=3.3, d=5.3):
        weights = self.dphidzdOmega(self.z_center, 100*u.TeV, a=a, b=b, c=c, d=d).value
        weights *= self.z_width
        weights /= weights.sum()
        return weights

    def calc_weights_flat(self):
        return self.calc_weights_SFR(a=1, b=0, c=np.inf, d=1)

    def calc_weights_cutoff_pl(self, m=3., k=-1.):
        weights = []
        for zlo, zhi in zip(self.z_center - self.z_width / 2, self.z_center + self.z_width / 2):
            zlo = max(zlo, 0.001)
            z = np.linspace(zlo, zhi, 2)
            dndz = (1 + z)**m * np.exp(z/k) * cosmo.differential_comoving_volume(z).value
            dfluxdz = dndz / cosmo.luminosity_distance(z).to(u.cm).value**2
            weights.append(trapz(dfluxdz, z))
        weights = np.array(weights) / np.sum(weights)
        return weights

    def minimize__lnL_SFR(self):
        # create multinorm object by side effect. TODO fix this
        for i in range(self.n_zbins):
            self.llhs[i].fit_bounds = None
            self.llhs[i].minimize__lnL_cov()

        def minfunc(params):
            ns, a, b, c, d, gamma = params
            llh = self.log_likelihood_SFR(ns, a, b, c, d, gamma)
            return -llh

        initial = [1500, 0.00170, 0.13, 3.3, 5.3, 2.5]
        bounds = [[0, None], [0, None], [0, None], [0, None], [0, None], [1, 4]]
        soln = minimize(minfunc, initial, bounds=bounds, method='Nelder-Mead')
        return soln.x, - 2 * (minfunc(soln.x) - minfunc([0, 0.00170, 0.13, 3.3, 5.3, 2.5]))

    def log_likelihood_SFR(self, ns, a, b, c, d, gamma):
        weights = self.calc_weights_SFR(a, b, c, d)
        nss = ns * weights
        llh = 0
        for i in range(self.n_zbins):
            llh += self.llhs[i].log_likelihood_ns_gamma(nss[i], gamma)
        return llh

    def BIC_SFR(self, ns, a, b, c, d, gamma):
        n_obs = self.n_zbins * Defaults.NEbin * (Defaults.NCL - self.lmin)
        return 6 * np.log(n_obs) - 2 * self.log_likelihood_SFR(ns, a, b, c, d, gamma)

    def BIC_null(self):
        return -2 * self.log_likelihood_linked(0, 2.5)

    def log_likelihood_cutoff_pl(self, ns, m, k, gamma):
        weights = self.calc_weights_cutoff_pl(m, k)
        nss = ns * weights
        llh = 0
        for i in range(self.n_zbins):
            llh += self.llhs[i].log_likelihood_ns_gamma(nss[i], gamma)
        return llh

    def minimize__lnL_cutoff_pl(self):
        # create multinorm object by side effect. TODO fix this
        for i in range(self.n_zbins):
            self.llhs[i].fit_bounds = None
            self.llhs[i].minimize__lnL_cov()

        def minfunc(params):
            ns, m, k, gamma = params
            llh = self.log_likelihood_cutoff_pl(ns, m, k, gamma)
            return -llh

        initial = [10000, self.m, self.k, self.gamma]
        bounds = [[0, None], [-10, 10], [-10, 10], [1, 4]]
        soln = minimize(minfunc, initial, bounds=bounds, method='Nelder-Mead')
        return soln.x, - 2 * (minfunc(soln.x) - minfunc([0, 3., -1., 2.5]))

    def BIC_cutoff_pl(self, ns, m, k, gamma):
        n_obs = self.n_zbins * Defaults.NEbin * (Defaults.NCL - self.lmin)
        return 4 * np.log(n_obs) - 2 * self.log_likelihood_cutoff_pl(ns, m, k, gamma)
