import itertools

from .internal import HoomdContext
import hoomd
import flowws
import numpy as np

class Init(flowws.Stage):
    ARGS = list(itertools.starmap(
        flowws.Stage.ArgumentSpecification,
        [
            ('number', int, None, 'Number of particles to simulate'),
        ]
    ))

    def run(self, scope, storage):
        particle_number = self.arguments['number']
        grid_n = int(np.ceil(particle_number**(1./3)))
        grid_x = np.arange(grid_n) - 0.5*grid_n

        positions = np.array(list(itertools.product(
            grid_x, grid_x, grid_x)), dtype=np.float32)
        # randomly select particles if we created more than necessary
        if len(positions) != particle_number:
            select_indices = np.arange(len(positions))
            np.random.shuffle(select_indices)
            select_indices = select_indices[:particle_number]
            select_indices = np.sort(select_indices)
            positions = positions[select_indices]

        box = hoomd.data.boxdim(grid_n, grid_n, grid_n, 0, 0, 0)

        try:
            with HoomdContext(scope, storage) as ctx:
                return
        except FileNotFoundError:
            with HoomdContext(scope, storage, restore=False) as ctx:
                snapshot = hoomd.data.make_snapshot(particle_number, box)
                snapshot.particles.position[:] = positions
                system = hoomd.init.read_snapshot(snapshot)
