import copy

import csky as cy

from . import Defaults

def dataset_modifications_factory(ebinmin, ebinmax):
    def dataset_modifications(self, ds):
        sig = copy.deepcopy(ds.sig)
        data = copy.deepcopy(ds.data)

        elo = Defaults.map_logE_edge[ebinmin]
        ehi = Defaults.map_logE_edge[ebinmax]

        sig_idx = (ds.sig['log10energy'] > elo) * (ds.sig['log10energy'] < ehi)
        sig = ds.sig[sig_idx]
        data_idx = (ds.data['log10energy'] > elo) * (ds.data['log10energy'] < ehi)
        data = ds.data[data_idx]

        ds.sig, ds.data = sig, data

    return dataset_modifications


class data_spec_factory(object):
    def __init__(self, ebinmin=0, ebinmax=-1):
        self.ebinmin = ebinmin
        self.ebinmax = ebinmax
        dataset_modifications = dataset_modifications_factory(ebinmin, ebinmax)


        self.IC40 = cy.selections.PSDataSpecs.IC40
        self.IC59 = cy.selections.PSDataSpecs.IC59
        self.IC79 = cy.selections.PSDataSpecs.IC79
        self.IC86_2011 = cy.selections.PSDataSpecs.IC86_2011
        self.IC86v3_2012_2017 = cy.selections.PSDataSpecs.IC86v3_2012_2017
        self.IC86v4 = cy.selections.PSDataSpecs.IC86v4
        self.DNNCascade = cy.selections.DNNCascadeDataSpecs.DNNCascade_10yr
        self.ESTES_10yr = cy.selections.ESTESDataSpecs.ESTES_2011_2021

        self.IC40.dataset_modifications = dataset_modifications
        self.IC59.dataset_modifications = dataset_modifications
        self.IC79.dataset_modifications = dataset_modifications
        self.IC86_2011.dataset_modifications = dataset_modifications
        self.IC86v3_2012_2017.dataset_modifications = dataset_modifications
        self.IC86v4.dataset_modifications = dataset_modifications
        self.DNNCascade.dataset_modifications = dataset_modifications
        self.ESTES_10yr.dataset_modifications = dataset_modifications

        self.ps_3yr = [self.IC79, self.IC86_2011, self.IC86v3_2012_2017]
        self.ps_10yr = [self.IC40, self.IC59, self.IC79, self.IC86_2011, self.IC86v3_2012_2017]
        self.ps_v4 = [self.IC40, self.IC59, self.IC79, self.IC86v4]
        self.estes_10yr = [self.ESTES_10yr]

"""
        class IC40(cy.selections.PSDataSpecs.IC40):
            ebinmin, ebinmax = ebinmin, ebinmax
            def dataset_modifications(self, ds):
                ds.sig, ds.data = restrict_energy(ds)


        class IC59(cy.selections.PSDataSpecs.IC59):
            ebinmin, ebinmax = ebinmin, ebinmax
            def dataset_modifications(self, ds):
                ds.sig, ds.data = self.restrict_energy(ds)


        class IC79(cy.selections.PSDataSpecs.IC79):
            ebinmin, ebinmax = ebinmin, ebinmax
            def dataset_modifications(self, ds):
                ds.sig, ds.data = self.restrict_energy(ds)


        class IC86_2011(cy.selections.PSDataSpecs.IC86_2011):
            ebinmin, ebinmax = ebinmin, ebinmax
            def dataset_modifications(self, ds):
                ds.sig, ds.data = self.restrict_energy(ds)


        class IC86v3_2012_2017(cy.selections.PSDataSpecs.IC86v3_2012_2017):
            ebinmin, ebinmax = ebinmin, ebinmax
            def dataset_modifications(self, ds):
                ds.sig, ds.data = self.restrict_energy(ds)


        class IC86v4(cy.selections.PSDataSpecs.IC86v4):
            ebinmin, ebinmax = ebinmin, ebinmax
            def dataset_modifications(self, ds):
                ds.sig, ds.data = self.restrict_energy(ds)


        class DNNCascade(cy.selections.DNNCascadeDataSpecs.DNNCascade_10yr):
            ebinmin, ebinmax = ebinmin, ebinmax
            def dataset_modifications(self, ds):
                ds.sig, ds.data = self.restrict_energy(ds)


        class ESTES_10yr(cy.selections.ESTESDataSpecs.ESTES_2011_2021):
            ebinmin, ebinmax = ebinmin, ebinmax
            def data_set_modifications(self, ds):
                ds.sig, ds.data = self.restrict_energy(ds)

                


    def restrict_energy(self, ds):
        sig = copy.deepcopy(ds.sig)
        data = copy.deepcopy(ds.data)

        elo = Defaults.map_logE_edge[self.ebinmin]
        ehi = Defaults.map_logE_edge[self.ebinmax]

        sig_idx = (ds.sig['log10energy'] > elo) * (ds.sig['log10energy'] < ehi)
        sig = ds.sig[sig_idx]
        data_idx = (ds.data['log10energy'] > elo) * (ds.data['log10energy'] < ehi)
        data = ds.data[data_idx]

        return sig, data
"""
