"""Class to generate events spanning multiple redshift bins"""

import numpy as np
import csky as cy

from . import Defaults
from .NeutrinoSample import NeutrinoSample


class TomographicEventGenerator(object):

    def __init__(self, event_generators, relative_weights):
        """
        Parameters
        ----------
        event_generators : list of EventGenerator
            List of CskyEventGenerators
        relative_weights : list of float
            Relative weights of each event generator.
        """

        if np.abs(np.sum(relative_weights) - 1) > 1e-6:
            raise ValueError("Relative weights must sum to 1")
        if len(event_generators) != len(relative_weights):
            raise ValueError("Length of event_generators and relative_weights must be the same")

        self.event_generators = event_generators
        self.relative_weights = np.array(relative_weights)

    def SyntheticTrial(self, n_events):
        """
        Parameters
        ----------
        n_events : int
            Number of events to generate
        """

        n_events_per_bin = (n_events * self.relative_weights).astype(int)

        for i, eg in enumerate(self.event_generators):
            if n_events > 0:
                if i == 0:
                    trial, nexc = eg.SyntheticTrial(n_events_per_bin[i])
                else:
                    if n_events_per_bin[i] > 0:
                        next_trial, nexc = eg.SyntheticTrial(n_events_per_bin[i])
                        for j in range(len(trial)):
                            next_trial[j][1]['energy'] = 10 ** next_trial[j][1]['log10energy']
                            trial[j][1]['energy'] = 10 ** trial[j][1]['log10energy']
                            trial[j][1] = cy.utils.Events.concatenate((trial[j][1], next_trial[j][1]))
            else:
                trial, nexc = self.event_generators[0].SyntheticTrial(n_events)

        return trial, 0
