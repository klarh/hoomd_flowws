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
class Dshemuchadse2021LJGInteraction(Dshemuchadse2021InteractionBase):
    """Specify a new interaction potential from the paper "Moving beyond the constraints of chemistry via crystal structure discovery with isotropic multiwell pair potentials" to include in future MD stages

    These interactions are taken from the methods description in the
    paper (Proceedings of the National Academy of Sciences May 2021,
    118 (21); DOI 10.1073/pnas.2024034118). This module implements the
    Lennard-Jones Gauss potential, consisting of a Lennard-Jones
    interaction plus a Gaussian.

    The potential is rescaled such that the global minimum is -1
    epsilon_0.

    """
    ARGS = [
        Arg('reset', '-r', bool, False,
            help='Disable previously-defined interactions'),
        Arg('epsilon', '-e', float,
            help='Attractive depth of the Gaussian interaction'),
        Arg('r_0', type=float,
            help='Gaussian center location'),
        Arg('sigma_squared_gaussian', type=float, default=.02,
            help='Parameter controlling width of the Gaussian'),
        Arg('width', '-w', int, 1000,
            help='Number of points at which to evaluate the tabulated potential'),
        Arg('r_min', None, float, .5,
            help='Minimum distance at which to evaluate the tabulated potential'),
        Arg('r_max', None, float, 2.5,
            help='Maximum distance at which to evaluate the tabulated potential'),
    ]

    def run(self, scope, storage):
        """Registers this object to provide a force compute in future MD stages"""
        callbacks = scope.setdefault('callbacks', collections.defaultdict(list))

        if self.arguments['reset']:
            pre_run_callbacks = [c for c in callbacks['pre_run']
                                 if not isinstance(c, Dshemuchadse2021LJGInteraction)]
            callbacks['pre_run'] = pre_run_callbacks
            return

        self.potential_kwargs = dict(
            epsilon=self.arguments['epsilon'], r_0=self.arguments['r_0'],
            sigma_sq=self.arguments['sigma_squared_gaussian'])
        (_, self.potential_kwargs['scale']) = self.find_potential_parameters()
        self.rmax = self.arguments['r_max']

        callbacks['pre_run'].append(self)
        scope.setdefault('visuals', []).append(self)

    @staticmethod
    def force(r, epsilon, r_0, sigma_sq, scale=1.):
        result = (12*r**(-13) - 12*r**(-7) -
                  epsilon*(r - r_0)/sigma_sq*np.exp(-(r - r_0)**2/2/sigma_sq))
        return result*scale

    @staticmethod
    def potential(r, epsilon, r_0, sigma_sq, scale=1.):
        result = r**(-12) - 2*r**(-6) - epsilon*np.exp(-(r - r_0)**2/2/sigma_sq)
        return result*scale
