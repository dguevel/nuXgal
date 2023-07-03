"""Classes for signal and background models"""

import os

from . import Defaults
from .CskyEventGenerator import CskyEventGenerator
from .NeutrinoSample import NeutrinoSample

import numpy as np
from tqdm import tqdm


class Model(object):
    """Base class for signal and background models"""

    def __init__(self,
                 galaxy_sample,
                 N_yr,
                 idx_mask=None,
                 save_model=True,
                 recompute=False):
        self.galaxy_sample = galaxy_sample
        self.name = galaxy_sample.galaxyName
        self.N_yr = N_yr
        self.Ebinmin = 0
        self.Ebinmax = -1
        self.idx_mask = idx_mask
        self.pretty_name = '{galaxyName}-{nyear}yr'.format(
            galaxyName=self.name,
            nyear=self.N_yr)

        # define filenames
        self.w_mean_fname = Defaults.SYNTHETIC_W_MEAN_FORMAT.format(
            galaxyName=self.name,
            nyear=self.N_yr)
        self.w_std_fname = Defaults.SYNTHETIC_W_STD_FORMAT.format(
            galaxyName=self.name,
            nyear=self.N_yr)
        self.w_atm_std_fname = Defaults.SYNTHETIC_ATM_CROSS_CORR_STD_FORMAT
        self.w_atm_std_fname = self.w_atm_std_fname.format(
            galaxyName=self.name,
            nyear=self.N_yr)

        # try to load model from file if it exists
        mean_exists = os.path.exists(self.w_mean_fname)
        std_exists = os.path.exists(self.w_std_fname)
        atm_std_exists = os.path.exists(self.w_atm_std_fname)
        files_exist = mean_exists and std_exists and atm_std_exists
        if recompute or not files_exist:
            self.calc_w_mean(N_re=500)
            self.calc_w_atm_std(N_re=500)
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
        np.save(self.w_atm_std_fname, self.w_atm_std)

    def load_model(self):
        self.w_mean = np.load(self.w_mean_fname)
        self.w_std = np.load(self.w_std_fname)
        self.w_atm_std = np.load(self.w_atm_std_fname)

    def get_event_generator(self):
        if not hasattr(self, 'event_generator'):
            self.event_generator = CskyEventGenerator(
                self.N_yr,
                self.galaxy_sample,
                gamma=self.gamma,
                Ebinmin=self.Ebinmin,
                Ebinmax=self.Ebinmax,
                idx_mask=self.idx_mask)

        return self.event_generator

    def calc_w_atm_std(self, N_re=500):
        w_cross = np.zeros((N_re, Defaults.NEbin, 3 * Defaults.NSIDE))
        ns = NeutrinoSample()
        eg = self.get_event_generator()

        for iteration in tqdm(np.arange(N_re)):

            trial, _ = eg.SyntheticTrial(0, self.idx_mask)
            ns.inputTrial(trial)
            ns.updateMask(self.idx_mask)
            w_cross[iteration] = ns.getCrossCorrelation(self.galaxy_sample)

        self.w_atm_trials = w_cross.copy()
        self.w_atm_mean = np.mean(w_cross, axis=0)
        self.w_atm_std = np.std(w_cross, axis=0)


class TemplateModel(Model):

    def calc_w_mean(self, N_re=500):
        w_cross = np.zeros((N_re, Defaults.NEbin, 3 * Defaults.NSIDE))
        ns = NeutrinoSample()
        eg = self.get_event_generator()

        for iteration in tqdm(np.arange(N_re)):

            trial, _ = eg.SyntheticTrial(1000000,
                                         self.idx_mask,
                                         signal_only=True)
            ns.inputTrial(trial)
            ns.updateMask(self.idx_mask)
            w_cross[iteration] = ns.getCrossCorrelation(self.galaxy_sample)

        self.w_trials = w_cross.copy()
        self.w_mean = np.mean(w_cross, axis=0)
        self.w_std = np.std(w_cross, axis=0)


class TemplateSignalModel(TemplateModel):
    gamma = 2.5


class TemplateBackgroundModel(TemplateModel):
    gamma = 3.7


class DataScrambleBackgroundModel(Model):
    def __init__(self):
        raise NotImplementedError(
            'DataScrambleBackgroundModel not implemented yet')
    gamma = 2.5
