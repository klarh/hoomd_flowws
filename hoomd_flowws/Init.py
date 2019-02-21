import itertools

from .internal import HoomdContext
import hoomd
import hoomd.dem
import flowws
import numpy as np

class Init(flowws.Stage):
    ARGS = list(itertools.starmap(
        flowws.Stage.ArgumentSpecification,
        [
            ('number', int, None, 'Number of particles to simulate'),
            ('mass_scale', float, 1, 'Scaling factor for mass of all particles'),
        ]
    ))

    def run(self, scope, storage):
        # factor to scale initial particle distance by
        spacing = 1.

        if 'type_shapes' in scope:
            shapes = scope['type_shapes']
            spacing = 2*max(shape.get('circumsphere_radius', 0.5) for shape in shapes)

            type_moments = []
            for shape in scope['type_shapes']:
                vertices = shape['vertices']
                if len(vertices[0]) == 2:
                    (_, _, inertia_tensor) = hoomd.dem.utils.massProperties(vertices)
                    type_moments.append(
                        (inertia_tensor[0], inertia_tensor[3], 0))
                else:
                    (vertices, faces) = hoomd.dem.utils.convexHull(vertices)
                    (_, _, inertia_tensor) = hoomd.dem.utils.massProperties(
                        vertices, faces)
                    type_moments.append(
                        (inertia_tensor[0], inertia_tensor[3], inertia_tensor[5]))
            type_moments = np.array(type_moments, dtype=np.float32)
        else:
            type_moments = None

        particle_number = self.arguments['number']
        grid_n = int(np.ceil(particle_number**(1./3)))
        grid_x = (np.arange(grid_n) - 0.5*grid_n)*spacing

        positions = np.array(list(itertools.product(
            grid_x, grid_x, grid_x)), dtype=np.float32)
        # randomly select particles if we created more than necessary
        if len(positions) != particle_number:
            select_indices = np.arange(len(positions))
            np.random.shuffle(select_indices)
            select_indices = select_indices[:particle_number]
            select_indices = np.sort(select_indices)
            positions = positions[select_indices]

        box = hoomd.data.boxdim(grid_n*spacing, grid_n*spacing, grid_n*spacing, 0, 0, 0)

        try:
            with HoomdContext(scope, storage) as ctx:
                return
        except FileNotFoundError:
            with HoomdContext(scope, storage, restore=False) as ctx:
                snapshot = hoomd.data.make_snapshot(particle_number, box)
                snapshot.particles.position[:] = positions
                snapshot.particles.mass[:] *= self.arguments['mass_scale']

                if type_moments is not None:
                    moments = type_moments[snapshot.particles.typeid]
                    snapshot.particles.moment_inertia[:] = moments

                system = hoomd.init.read_snapshot(snapshot)
