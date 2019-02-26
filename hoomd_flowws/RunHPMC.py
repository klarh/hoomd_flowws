import collections
import itertools
import logging

import numpy as np

from .internal import intfloat, HoomdContext
import flowws
from flowws import Argument as Arg
import hoomd
import hoomd.hpmc
from . import Run

logger = logging.getLogger(__name__)

class RunHPMC(Run):
    ARGS = [
        Arg('steps', '-s', intfloat, None, required=True,
            help='Number of timesteps to run'),
        Arg('integrator', '-i', str, None, required=True,
            help='Integrator type'),
        Arg('pressure', None, float,
            help='Pressure for isobaric simulations'),
        Arg('backup_period', '-b', intfloat, 0,
            help='Period for dumping a backup file'),
        Arg('dump_period', '-d', intfloat, 0,
            help='Period for dumping a trajectory file'),
        Arg('expand_by', None, float,
            help='Expand each dimension of the box by this ratio during this stage'),
        Arg('compress_to', None, float,
            help='Compress to the given packing fraction during this stage (overrides expand_by)'),
        Arg('integrator_seed', None, int, 14,
            help='Random number seed for integration method'),
    ]

    def setup_integrator(self, scope, storage):
        system = scope['system']
        particle_types = system.particles.types
        integrator_type = self.arguments['integrator']

        type_shapes = scope.get('type_shapes', dict(type='Sphere'))
        shape_types = [shape['type'] for shape in type_shapes]
        rounding_radii = [shape.get('rounding_radius', 0) for shape in type_shapes]
        assert len(set(shape_types)) == 1, 'HPMC only supports systems with a single shape type (for example, all convex polyhedra)'
        shape_type = shape_types[0]

        if shape_type == 'Sphere':
            integrator = hoomd.hpmc.integrate.sphere(self.arguments['integrator_seed'])

        elif shape_type == 'Polygon':
            if max(rounding_radii) > 0:
                assert all(not is_concave(shape) for shape in type_shapes), 'Rounded polygons must be convex'
                integrator = hoomd.hpmc.integrate.convex_spheropolygon(
                    self.arguments['integrator_seed'])
            else:
                if any(is_concave(shape) for shape in type_shapes):
                    integrator = hoomd.hpmc.integrate.simple_polygon(
                        self.arguments['integrator_seed'])
                else:
                    integrator = hoomd.hpmc.integrate.convex_polygon(
                        self.arguments['integrator_seed'])

            for (typ, shape) in zip(particle_types, type_shapes):
                integrator.shape_param.set(
                    typ, vertices=shape['vertices'],
                    sweep_radius=shape.get('rounding_radius', 0))

        elif shape_type == 'ConvexPolyhedron':
            if max(rounding_radii) > 0:
                integrator = hoomd.hpmc.integrate.convex_spheropolyhedron(
                    self.arguments['integrator_seed'])
            else:
                integrator = hoomd.hpmc.integrate.convex_polyhedron(
                    self.arguments['integrator_seed'])

            for (typ, shape) in zip(particle_types, type_shapes):
                integrator.shape_param.set(
                    typ, vertices=shape['vertices'],
                    sweep_radius=shape.get('rounding_radius', 0))

        if integrator_type == 'nvt':
            pass # integrators are nvt by default
        else:
            raise NotImplementedError(integrator_type)

        return integrator

def is_concave(polygon):
    vertices = np.array(polygon['vertices'], dtype=np.float32)

    delta = np.roll(vertices, -1, axis=0) - vertices
    return np.any(np.cross(delta, np.roll(delta, -1, axis=0)) < 0)
