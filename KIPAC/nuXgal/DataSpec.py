import copy

import csky as cy

from . import Defaults

def restrict_energy(ds):
    sig = copy.deepcopy(ds.sig)
    data = copy.deepcopy(ds.data)

    elo = Defaults.map_logE_edge.min()
    ehi = Defaults.map_logE_edge.max()

    sig = ds.sig[(ds.sig['log10energy'] > elo) * (ds.sig['log10energy'] < ehi)]
    data = ds.data[(ds.data['log10energy'] > elo) * (ds.data['log10energy'] < ehi)]
    return sig, data

class IC40(cy.selections.PSDataSpecs.IC40):
    def dataset_modifications(self, ds):
        ds.sig, ds.data = restrict_energy(ds)
class IC59(cy.selections.PSDataSpecs.IC59):
    def dataset_modifications(self, ds):
        ds.sig, ds.data = restrict_energy(ds)
class IC79(cy.selections.PSDataSpecs.IC79):
    def dataset_modifications(self, ds):
        ds.sig, ds.data = restrict_energy(ds)
class IC86_2011(cy.selections.PSDataSpecs.IC86_2011):
    def dataset_modifications(self, ds):
        ds.sig, ds.data = restrict_energy(ds)
class IC86v3_2012_2017(cy.selections.PSDataSpecs.IC86v3_2012_2017):
    def dataset_modifications(self, ds):
        ds.sig, ds.data = restrict_energy(ds)

ps_3yr = [IC79, IC86_2011, IC86v3_2012_2017]
ps_10yr = [IC40, IC59, IC79, IC86_2011, IC86v3_2012_2017]