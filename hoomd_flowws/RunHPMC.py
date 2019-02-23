import collections
import itertools
import logging

import numpy as np

from .internal import intfloat, HoomdContext
import flowws
import hoomd
import hoomd.hpmc
from . import Run

logger = logging.getLogger(__name__)

class RunHPMC(Run):
    ARGS = list(itertools.starmap(
        flowws.Stage.ArgumentSpecification,
        [
            ('steps', intfloat, None, 'Number of timesteps to run'),
            ('integrator', str, None, 'Integrator type'),
            ('pressure', float, 1, 'Pressure for isobaric simulations'),
            ('backup_period', intfloat, 0, 'Period for dumping a backup file'),
            ('dump_period', intfloat, 0, 'Period for dumping a trajectory file'),
            ('expand_by', float, None, 'Expand each dimension of the box by this ratio during this stage'),
            ('compress_to', float, None, 'Compress to the given packing fraction during this stage (overrides expand_by)'),
            ('integrator_seed', int, 14, 'Random number seed for integration method'),
        ]
    ))

    def setup_integrator(self, scope, storage):
        system = scope['system']
        particle_types = system.particles.types
        integrator_type = self.arguments['integrator']

        type_shapes = scope.get('type_shapes', dict(type='Sphere'))
        shape_types = [shape['type'] for shape in type_shapes]
        rounding_radii = [shape.get('rounding_radius', 0) for shape in type_shapes]
        assert len(set(shape_types)) == 1, 'HPMC only supports systems with a single shape type'
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
