import collections
import itertools
import logging

import numpy as np

import hoomd.dem
import flowws
from flowws import Argument as Arg

logger = logging.getLogger(__name__)

@flowws.add_stage_arguments
class DEMInteraction(flowws.Stage):
    """Specify that DEM interactions should be included in future MD stages"""
    ARGS = [
        Arg('reset', '-r', bool, False,
            help='Clear previously-defined DEM interactions beforehand'),
        Arg('type', '-t', str, required=True,
            help='Interaction class name'),
    ]

    def run(self, scope, storage):
        """Registers this object to provide a DEM force compute in future MD stages"""
        callbacks = scope.setdefault('callbacks', collections.defaultdict(list))

        if self.arguments['reset']:
            pre_run_callbacks = [c for c in callbacks['pre_run']
                                 if not isinstance(c, DEMInteraction)]
            callbacks['pre_run'] = pre_run_callbacks

        callbacks['pre_run'].append(self)

    def __call__(self, scope, storage, context):
        """Callback to be performed before each run command.

        Initializes a DEM pair potential interaction based on per-type
        shape information.
        """
        interaction_type = self.arguments['type']

        nlist = hoomd.md.nlist.tree()
        system = scope['system']
        dimensions = scope.get('dimensions', 3)

        try:
            type_shapes = scope['type_shapes']
        except KeyError:
            msg = ('Shape information has not been set for DEM interactions. '
                   'Use a ShapeDefinition or similar step beforehand.')
            raise WorkflowError(msg)

        if interaction_type == 'wca':
            radii = [shape.get('rounding_radius', 0) for shape in type_shapes]
            assert np.isclose(min(radii), max(radii)), 'WCA requires identical rounding radii for all shapes'
            radius = radii[0]

            if radius <= 0:
                logger.warning('Non-rounded shapes given, using a rounding radius of 0.5')
                radius = .5

            potential = hoomd.dem.pair.WCA(nlist, radius)
            for (name, shape) in zip(system.particles.types, type_shapes):
                vertices = shape['vertices']
                if dimensions == 2:
                    potential.setParams(name, vertices)
                else:
                    (vertices, faces) = hoomd.dem.utils.convexHull(shape['vertices'])
                    potential.setParams(name, vertices, faces)
        else:
            raise NotImplementedError(
                'Unknown Interaction type {}'.format(interaction_type))
