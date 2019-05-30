import itertools

from .internal import HoomdContext, intfloat
import hoomd
import hoomd.dem
import flowws
from flowws import Argument as Arg
import numpy as np

@flowws.add_stage_arguments
class Init(flowws.Stage):
    """Initialize a system

    Currently simply places points on a simple cubic lattice.
    """
    ARGS = [
        Arg('number', '-n', intfloat, required=True,
            help='Number of particles to simulate'),
        Arg('mass_scale', None, float, 1,
            help='Scaling factor for mass of all particles'),
        Arg('type_ratios', '-t', [float],
            help='Prevalence (ratio) of each particle type'),
    ]

    def run(self, scope, storage):
        # factor to scale initial particle distance by
        spacing = 1.

        dimensions = scope.get('dimensions', 3)
        assert dimensions in (2, 3)

        # treat default density to give 1 m_0 per unit-diameter sphere
        particle_density = 4/np.pi if dimensions == 2 else 6/np.pi

        default_type_ratios = [1]*len(scope.get('type_shapes', [None]))
        type_ratios = np.array((self.arguments.get('type_ratios', []) or default_type_ratios), dtype=np.float32)
        type_ratios /= np.sum(type_ratios)

        if 'type_shapes' in scope:
            shapes = scope['type_shapes']

            assert len(type_ratios) == len(shapes)

            spacing = 2*max(shape.get('circumsphere_radius', 0.5) for shape in shapes)

            type_masses = []
            type_moments = []
            for shape in scope['type_shapes']:
                # "volume" here is an area in 2D
                volume = np.polyval(
                    shape['rounding_volume_polynomial'], shape.get('rounding_radius', 0))
                type_masses.append(particle_density*volume)

                # approximate increase in moment of inertia based on
                # rounded vs non-rounded volume
                moment_adjustment = volume/np.polyval(shape['rounding_volume_polynomial'], 0)

                vertices = shape['vertices']

                if len(vertices[0]) == 2:
                    (_, _, inertia_tensor) = hoomd.dem.utils.massProperties(vertices)
                    moment = np.array((0, 0, inertia_tensor[5]))
                    type_moments.append(moment*moment_adjustment)
                else:
                    (vertices, faces) = hoomd.dem.utils.convexHull(vertices)
                    (_, _, inertia_tensor) = hoomd.dem.utils.massProperties(
                        vertices, faces)
                    moment = np.array((inertia_tensor[0], inertia_tensor[3], inertia_tensor[5]))
                    type_moments.append(moment*moment_adjustment)
        else:
            type_masses = len(type_ratios)*[1]
            type_moments = len(type_ratios)*[(0, 0, 0)]

        type_masses = (np.array(type_masses, dtype=np.float32)*
                       self.arguments['mass_scale'])
        type_moments = (np.array(type_moments, dtype=np.float32)*
                        particle_density*self.arguments['mass_scale'])

        type_names = [chr(ord('A') + i) for i in range(len(type_ratios))]

        particle_number = self.arguments['number']
        grid_n = int(np.ceil(particle_number**(1./dimensions)))
        grid_x = (np.arange(grid_n) - 0.5*grid_n)*spacing
        grid_z = [0] if dimensions == 2 else grid_x

        positions = np.array(list(itertools.product(
            grid_x, grid_x, grid_z)), dtype=np.float32)
        # randomly select particles if we created more than necessary
        if len(positions) != particle_number:
            select_indices = np.arange(len(positions))
            np.random.shuffle(select_indices)
            select_indices = select_indices[:particle_number]
            select_indices = np.sort(select_indices)
            positions = positions[select_indices]

        types = np.zeros(particle_number, dtype=np.int32)
        type_indices = (np.cumsum([0] + type_ratios.tolist())*particle_number).astype(np.uint32)
        for i, (start, end) in enumerate(zip(type_indices[:-1], type_indices[1:])):
            types[start:end] = i
        np.random.shuffle(types)

        Lz = 1 if dimensions == 2 else grid_n*spacing
        box = hoomd.data.boxdim(
            grid_n*spacing, grid_n*spacing, Lz, 0, 0, 0, dimensions=dimensions)

        try:
            with HoomdContext(scope, storage) as ctx:
                return
        except FileNotFoundError:
            with HoomdContext(scope, storage, restore=False) as ctx:
                snapshot = hoomd.data.make_snapshot(particle_number, box)

                snapshot.particles.position[:] = positions
                snapshot.particles.mass[:] = type_masses[types]
                snapshot.particles.moment_inertia[:] = type_moments[types]
                snapshot.particles.types = type_names
                snapshot.particles.typeid[:] = types

                system = hoomd.init.read_snapshot(snapshot)
