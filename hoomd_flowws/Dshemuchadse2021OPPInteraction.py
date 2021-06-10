import argparse
import collections
import logging
import io
import itertools

import flowws
from flowws import Argument as Arg
import numpy as np

from .dshemuchadse2021_internal import Dshemuchadse2021InteractionBase

logger = logging.getLogger(__name__)

@flowws.add_stage_arguments
class Dshemuchadse2021OPPInteraction(Dshemuchadse2021InteractionBase):
    """Specify a new interaction potential from the paper "Moving beyond the constraints of chemistry via crystal structure discovery with isotropic multiwell pair potentials" to include in future MD stages

    These interactions are taken from the methods description in the
    paper (Proceedings of the National Academy of Sciences May 2021,
    118 (21); DOI 10.1073/pnas.2024034118). This module implements the
    oscillatory pair potential, consisting of a short-range repulsion
    and a cosine term that scales with r^-3.

    The potential is rescaled such that the global minimum is -1
    epsilon_0.

    """
    ARGS = [
        Arg('reset', '-r', bool, False,
            help='Disable previously-defined interactions'),
        Arg('k', '-k', float,
            help='Interaction parameter k'),
        Arg('phi', '-p', float,
            help='Interaction parameter phi'),
        Arg('width', '-w', int, 1000,
            help='Number of points at which to evaluate the tabulated potential'),
        Arg('r_min', None, float, .5,
            help='Minimum distance at which to evaluate the tabulated potential'),
    ]

    def run(self, scope, storage):
        """Registers this object to provide a force compute in future MD stages"""
        callbacks = scope.setdefault('callbacks', collections.defaultdict(list))

        if self.arguments['reset']:
            pre_run_callbacks = [c for c in callbacks['pre_run']
                                 if not isinstance(c, Dshemuchadse2021OPPInteraction)]
            callbacks['pre_run'] = pre_run_callbacks
            return

        self.potential_kwargs = dict(k=self.arguments['k'], phi=self.arguments['phi'])
        self.rmax, self.potential_kwargs['scale'] = self.find_potential_parameters()

        callbacks['pre_run'].append(self)
        scope.setdefault('visuals', []).append(self)

    @staticmethod
    def force(r, k, phi, scale=1.):
        arg = k*(r - 1.) + phi
        result = 15*r**(-16) + (k*r*np.sin(arg) + 3*np.cos(arg))*r**(-4)
        return result*scale

    @staticmethod
    def potential(r, k, phi, scale=1.):
        result = r**(-15) + np.cos(k*(r - 1.) + phi)*r**(-3)
        return result*scale
