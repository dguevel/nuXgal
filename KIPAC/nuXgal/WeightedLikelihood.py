import numpy as np
import os

from KIPAC.nuXgal import Defaults
from KIPAC.nuXgal.Likelihood import Likelihood
from KIPAC.nuXgal.WeightedNeutrinoSample import WeightedNeutrinoSample


class WeightedLikelihood(Likelihood):
    BlurredGalaxyMapFname = Defaults.WEIGHTED_BLURRED_GALAXYMAP_FORMAT
    AtmSTDFname = Defaults.WEIGHTED_SYNTHETIC_ATM_CROSS_CORR_STD_FORMAT
    AtmNcountsFname = Defaults.WEIGHTED_SYNTHETIC_ATM_NCOUNTS_FORMAT

    def __init__(self, *args, **kwargs):
        Likelihood.__init__(self, *args, **kwargs)
        self.BlurredGalaxyMapFname = self.BlurredGalaxyMapFname.format(galaxyName=self.gs.galaxyName)
        self.AtmSTDFname = self.AtmSTDFname.format(galaxyName=self.gs.galaxyName, nyear= str(self.N_yr))
        self.AtmNcountsFname = self.AtmNcountsFname.format(galaxyName=self.gs.galaxyName, nyear= str(self.N_yr))
        self.neutrino_sample_class = WeightedNeutrinoSample

    def getPDFRatioWeight(self, subana, sig, gamma):
        pdf_ratio = subana.energy_pdf_ratio_model(sig)(gamma=gamma)[1]
        return pdf_ratio / (1 + pdf_ratio)


    def weighted_f_to_f(self, weighted_f, gamma):
        """Convert weighted fraction f to unweighted f. 
        The conversion is based on empirical scaling 
        relation from MC and atmospheric data"""

        alpha = -0.91549942
        beta = 1.1085688
        return weighted_f * gamma ** -alpha * np.exp(-beta)

    def f_to_weighted_f(self, f, gamma):
        """Convert weighted fraction f to unweighted f. 
        The conversion is based on empirical scaling 
        relation from MC and atmospheric data"""

        alpha = -0.91549942
        beta = 1.1085688
        return f * gamma ** alpha * np.exp(beta)