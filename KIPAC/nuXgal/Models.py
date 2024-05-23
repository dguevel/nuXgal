"""Classes for signal and background models"""

import os

from . import Defaults
from .CskyEventGenerator import CskyEventGenerator
from .NeutrinoSample import NeutrinoSample

import csky as cy
import numpy as np
from tqdm import tqdm
import healpy as hp
from scipy.stats import gaussian_kde

try:
    from classy import Class
except ImportError:
    print('classy not installed, cannot compute analytic models but can load')


class Model(object):
    """Base class for signal and background models"""

    def __init__(self,
                 galaxy_sample,
                 N_yr,
                 idx_mask=None,
                 save_model=True,
                 recompute=False,
                 gamma=2.5,
                 path_sig=''):
        self.galaxy_sample = galaxy_sample
        self.name = galaxy_sample.galaxyName
        self.N_yr = N_yr
        self.Ebinmin = 0
        self.Ebinmax = -1
        self.idx_mask = idx_mask
        self.gamma = gamma
        self.path_sig = path_sig
        self.pretty_name = '{galaxyName}-{nyear}yr'.format(
            galaxyName=self.name,
            nyear=self.N_yr)
        
        path_sig_tail = os.path.split(self.path_sig)[-1].split('.')[0]
        if len(path_sig_tail) > 0:
            path_sig_tail = '_' + path_sig_tail

        # define filenames
        self.w_mean_fname = Defaults.SYNTHETIC_W_MEAN_FORMAT.format(
            galaxyName=self.name,
            nyear=self.N_yr + path_sig_tail,
            method=self.method_type,
            gamma=self.gamma)
        self.w_std_fname = Defaults.SYNTHETIC_W_STD_FORMAT.format(
            galaxyName=self.name,
            nyear=self.N_yr + path_sig_tail,
            method=self.method_type,
            gamma=self.gamma)
        # self.w_atm_std_fname = Defaults.SYNTHETIC_ATM_CROSS_CORR_STD_FORMAT
        # self.w_atm_std_fname = self.w_atm_std_fname.format(
        #    galaxyName=self.name,
        #    nyear=self.N_yr)

        # try to load model from file if it exists
        mean_exists = os.path.exists(self.w_mean_fname)
        std_exists = os.path.exists(self.w_std_fname)
        files_exist = mean_exists and std_exists
        if recompute or not files_exist:
            self.calc_w_mean(N_re=1000)
            if save_model:
                self.save_model()
        else:
            self.load_model()

    def __repr__(self):
        return self.pretty_name

    def __str__(self):
        return self.pretty_name

    def save_model(self):
        np.save(self.w_mean_fname, self.w_mean)
        np.save(self.w_std_fname, self.w_std)

    def load_model(self):
        self.w_mean = np.load(self.w_mean_fname)
        self.w_std = np.load(self.w_std_fname)

    def get_event_generator(self, mc_background=False):
        if not hasattr(self, 'event_generator'):
            self.event_generator = CskyEventGenerator(
                self.N_yr,
                self.galaxy_sample,
                gamma=self.gamma,
                Ebinmin=self.Ebinmin,
                Ebinmax=self.Ebinmax,
                idx_mask=self.idx_mask,
                mc_background=mc_background,
                path_sig=self.path_sig)

        return self.event_generator
    

class GalaxyModel(Model):
    method_type = 'galaxy'

    def calc_w_mean(self, N_re=500):
        fsky = 1 - len(self.idx_mask[0]) / Defaults.NPIXEL
        w_cross = hp.anafast(self.galaxy_sample.overdensity, lmax=Defaults.MAX_L) / fsky
        self.w_mean = np.array([w_cross for i in range(Defaults.NEbin)]) - 4 * np.pi * fsky / self.galaxy_sample.galaxymap.sum()
        self.w_std = np.zeros_like(self.w_mean)


class TemplateModel(Model):
    method_type = 'template'

    def calc_w_mean(self, N_re=500):
        w_cross = np.zeros((N_re, Defaults.NEbin, 3 * Defaults.NSIDE))
        ns = NeutrinoSample()
        eg = self.get_event_generator()

        for iteration in tqdm(np.arange(N_re)):

            trial, _ = eg.SyntheticTrial(1000000,
                                         keep_total_constant=False,
                                         signal_only=True)
            ns.inputTrial(trial)
            ns.updateMask(self.idx_mask)
            w_cross[iteration] = ns.getCrossCorrelation(self.galaxy_sample)

        self.w_trials = w_cross.copy()
        self.w_mean = np.mean(w_cross, axis=0)
        self.w_std = np.std(w_cross, axis=0)


class AnalyticSignalModel(Model):
    method_type = 'CLASS_analytic'
    gamma = 2.5

    def calc_w_mean(self, N_re=500, estimator='anafast', ana=None):
        """Compute the mean cross-correlation function for the
        signal model using analytic methods."""
        # create instance of the class "Class"
        LambdaCDM = Class()

        # pass input parameters
        LambdaCDM.set({
            'omega_b': 0.0223828,
            'omega_cdm': 0.1201075,
            'h': 0.67810,
            'A_s': 2.100549e-09,
            'n_s': 0.9660499,
            'tau_reio': 0.05430842,
            'output': 'tCl,pCl,lCl,mPk,nCl',
            'lensing': 'yes',
            'P_k_max_1/Mpc': 3.0,
            'selection': 'gaussian',
            'l_max_lss': Defaults.MAX_L,
            # best match fit to galaxy autocorrelation function
            # these parameters don't match measured dN/dz; idk why
            'selection_mean': 0.16,
            'selection_width': 0.01,
            'selection_bias': 0.9
        })

        # run class
        LambdaCDM.compute()

        # get overdensity Cls
        cls_dens = LambdaCDM.density_cl(Defaults.MAX_L)

        # load beam files
        self.bl = np.zeros((Defaults.NEbin, Defaults.NCL))
        self.bl_fnames = []
        for ebin in range(Defaults.NEbin):
            bl_fname = Defaults.BEAM_FNAME_FORMAT.format(year='ps_v4', ebin=ebin)
            self.bl_fnames.append(bl_fname)
            self.bl[ebin] = np.load(bl_fname)

        pixwin = hp.pixwin(Defaults.NSIDE)
        self.w_mean = cls_dens['dd'][0] * pixwin**2
        self.w_mean = np.array([self.w_mean * self.bl[i] for i in range(Defaults.NEbin)])
        self.w_std = np.zeros_like(self.w_mean) * np.nan
        self.w_std = np.array([self.w_std for i in range(Defaults.NEbin)])


class DataScrambleBackgroundModel(Model):
    method_type = 'data_scramble'
    gamma = 2.5

    def calc_w_mean(self, N_re=500, estimator='anafast', ana=None):
        w_cross = np.zeros((N_re, Defaults.NEbin, 3 * Defaults.NSIDE))
        ns = NeutrinoSample()
        eg = self.get_event_generator()

        for iteration in tqdm(np.arange(N_re)):

            trial, _ = eg.SyntheticTrial(0)
            ns.inputTrial(trial)
            ns.updateMask(self.idx_mask)
            if estimator == 'anafast':
                w_cross[iteration] = ns.getCrossCorrelation(self.galaxy_sample)
            elif estimator == 'polspice':
                w_cross[iteration] = ns.getCrossCorrelationPolSpice(self.galaxy_sample, ana)


        self.w_trials = w_cross.copy()
        self.w_mean = np.mean(w_cross, axis=0)
        self.w_std = np.std(w_cross, axis=0)


class MCScrambleBackgroundModel(Model):
    method_type = 'mc_scramble_background'
    gamma = 2.5

    def calc_w_mean(self, N_re=500, estimator='anafast', ana=None):
        from tqdm import tqdm

        self.w_mean = np.zeros((Defaults.NEbin, Defaults.NCL))
        nu_map = np.zeros((Defaults.NEbin, Defaults.NPIXEL))

        eg = self.get_event_generator(mc_background=True)

        for n in tqdm(np.arange(N_re)):
            for i, (elo, ehi) in enumerate(zip(Defaults.map_logE_edge[:-1], Defaults.map_logE_edge[1:])):

                ra = []
                dec = []
                weight = []
                for bg_inj in eg.trial_runner.bg_injs:
                    idx = (bg_inj.mc['log10energy'] >= elo) * (bg_inj.mc['log10energy'] < ehi)
                    delta_ra = np.random.uniform(0, 2*np.pi, len(bg_inj.mc['ra'][idx]))
                    ra.append(bg_inj.mc['ra'][idx] + delta_ra)
                    dec.append(bg_inj.mc['dec'][idx])
                    weight.append(bg_inj.probs[0][idx])

                ra = np.concatenate(ra)
                dec = np.concatenate(dec)
                weight = np.concatenate(weight)
                pixels = hp.ang2pix(Defaults.NSIDE, np.degrees(ra), np.degrees(dec), lonlat=True)
                nu_map[i] += np.histogram(pixels, bins=Defaults.NPIXEL, weights=weight)[0]

        ns = NeutrinoSample()
        ns.inputCountsmap(nu_map)
        ns.updateMask(self.idx_mask)
        self.w_mean = ns.getCrossCorrelation(self.galaxy_sample)

        self.w_std = np.zeros_like(self.w_mean)


class MCBackgroundModel(Model):
    method_type = 'mc_background'
    gamma = 2.5

    def calc_w_mean(self, N_re=500, estimator='anafast', ana=None):
        from tqdm import tqdm

        self.w_mean = np.zeros((Defaults.NEbin, Defaults.NCL))
        nu_map = np.zeros((Defaults.NEbin, Defaults.NPIXEL))

        eg = self.get_event_generator(mc_background=True)

        for i, (elo, ehi) in enumerate(zip(Defaults.map_logE_edge[:-1], Defaults.map_logE_edge[1:])):

            mc = []
            weight = []
            for bg_inj in eg.trial_runner.bg_injs:
                idx = (bg_inj.mc['log10energy'] >= elo) * (bg_inj.mc['log10energy'] < ehi)
                mc.append(bg_inj.mc['sindec'][idx])
                weight.append(bg_inj.probs[0][idx])

            mc = np.concatenate(mc)
            weight = np.concatenate(weight)
            interp = gaussian_kde(mc, weights=weight)

            npix = hp.nside2npix(Defaults.NSIDE)
            pixels = np.arange(npix)
            ra, dec = hp.pix2ang(Defaults.NSIDE, pixels, lonlat=True)
            udec = np.unique(dec)
            pixels = pixels.astype(float)

            for udeci in tqdm(udec):
                n = interp(np.sin(np.radians(udeci)))
                pixels[np.where(dec == udeci)] = n[0]
            nu_map[i] = pixels.copy()
                
        ns = NeutrinoSample()
        ns.inputCountsmap(nu_map)
        ns.updateMask(self.idx_mask)
        self.w_mean = ns.getCrossCorrelation(self.galaxy_sample)


        self.w_std = np.zeros_like(self.w_mean)


class DataHistogramBackgroundModel(Model):
    method_type = 'data_histogram'
    _bins = np.arange(-1, 1.1, 0.05)

    def _load_events(self):
        events = []

        path_list = []
        for spec in ps_v4:
            spec = spec()
            if not isinstance(spec.path_data, str):
                path_list.extend(spec.path_data)
            else:
                path_list.append(spec.path_data)

        for path in path_list:
            path = os.path.join(
                cy.selections.Repository.remote_root,
                path)
            path = path.format(version=Defaults.ANALYSIS_VERSION)
            evt = cy.utils.Arrays(np.load(path))
            events.append(evt)

        events = cy.utils.Arrays.concatenate(events)
        return events

    def _calc_events_per_ebin(self, events):
        events_per_ebin = [0 for i in range(Defaults.NEbin)]
        for ebin in range(Defaults.NEbin):
            idx = events['log10energy'] >= Defaults.map_logE_edge[ebin]
            idx *= events['log10energy'] < Defaults.map_logE_edge[ebin+1]
            events_per_ebin[ebin] += np.sum(idx)
        return events_per_ebin

    def _generate_synthetic_trial(self):
        # randomly choose event dec bin weighted by histogram
        probs = self.hist / self.hist.sum()

        events_list = []

        for ebin in range(Defaults.NEbin):
            sindec_bin_idx = np.random.choice(
                len(self.hist),
                p=probs,
                size=self.events_per_ebin[ebin])

            sindec = self.bin_center[sindec_bin_idx]
            events = cy.utils.Arrays({'sindec': sindec})

            # assign uniform random RA
            events['ra'] = np.random.uniform(0, 2*np.pi, len(events))

            # assign uniform random sin(Dec) within bin
            dsindec = (self._bins[1] - self._bins[0]) / 2
            events['sindec'] += np.random.uniform(
                -dsindec,
                dsindec,
                len(events))
            events['dec'] = np.arcsin(events['sindec'])

            # assign uniform random energy within log energy bin
            events['log10energy'] = np.random.uniform(
                Defaults.map_logE_edge[ebin],
                Defaults.map_logE_edge[ebin+1],
                len(events))
            events_list.append(events)
        return [[cy.utils.Arrays.concatenate(events_list)]]

    def _make_histogram(self):
        self.events = self._load_events()
        self.events['log10energy'] = self.events['logE']
        self.events_per_ebin = self._calc_events_per_ebin(self.events)

        hist, bins = np.histogram(
            np.sin(self.events['dec']),
            bins=self._bins)
        self.hist = hist
        self.bin_center = (bins[1:] + bins[:-1]) / 2

    def calc_w_mean(self, N_re=500):
        self._make_histogram()
        w_cross = np.zeros((N_re, Defaults.NEbin, 3 * Defaults.NSIDE))
        ns = NeutrinoSample()

        for iteration in tqdm(np.arange(N_re)):

            trial = self._generate_synthetic_trial()
            ns.inputTrial(trial)
            ns.updateMask(self.idx_mask)
            w_cross[iteration] = ns.getCrossCorrelation(self.galaxy_sample)

        self.w_trials = w_cross.copy()
        self.w_mean = np.mean(w_cross, axis=0)
        self.w_std = np.std(w_cross, axis=0)


class FlatBackgroundModel(Model):
    method_type = 'flat'
    gamma = 3.7

    def calc_w_mean(self, N_re=500):
        self.w_mean = np.zeros((Defaults.NEbin, Defaults.NCL))
        self.w_std = np.ones((Defaults.NEbin, Defaults.NCL))

    def load_model(self):
        self.calc_w_mean()
