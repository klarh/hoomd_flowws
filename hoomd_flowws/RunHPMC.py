import collections
import contextlib
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

@flowws.add_stage_arguments
class RunHPMC(Run):
    """Run for some number of steps using HPMC"""
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
        Arg('tune', '-t', (float, int, intfloat),
            metavar=('acceptance_ratio', 'epochs', 'steps'),
            help='Tune move distances to achieve the target acceptance ratio '
            'after updating a given number of epochs and running the given '
            'number of steps at each epoch')
    ]

    def setup_integrator(self, scope, storage, context):
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

                for (typ, shape) in zip(particle_types, type_shapes):
                    integrator.shape_param.set(
                        typ, vertices=shape['vertices'],
                        sweep_radius=shape.get('rounding_radius', 0))
            else:
                integrator = hoomd.hpmc.integrate.convex_polyhedron(
                    self.arguments['integrator_seed'])

                for (typ, shape) in zip(particle_types, type_shapes):
                    integrator.shape_param.set(
                        typ, vertices=shape['vertices'])

        if integrator_type == 'nvt':
            pass # integrators are nvt by default
        else:
            raise NotImplementedError(integrator_type)

        frame, _ = self.load_move_distance(scope, storage, context, integrator)
        # we should only create a tuner object if we haven't already
        # done tuning for this stage
        can_tune = frame < scope['cumulative_steps'] - self.arguments['steps']

        if self.arguments.get('tune', None) and can_tune:
            import gtar # throw ImportErrors before running any steps
            (acceptance_ratio, epochs, period) = self.arguments['tune']

            child_arguments = dict(self.arguments)
            child_arguments['steps'] = epochs*period
            del child_arguments['tune']
            child_arguments['tune_acceptance_ratio'] = acceptance_ratio
            child_arguments['tune_epochs'] = epochs

            child_scope = dict(scope)
            child_scope['cumulative_steps'] += epochs*period - self.arguments['steps']

            TuneHPMC(**child_arguments).run(child_scope, storage)

            # now the tuned move distances have been saved, so we can
            # retrieve them for this stage
            self.load_move_distance(scope, storage, context, integrator)

        scope['integrator'] = integrator
        return integrator

    def load_move_distance(self, scope, storage, context, integrator):
        import gtar

        # per-type move distance arrays for translation/rotation/box
        distances = {}

        dump_filename = scope.get('dump_filename', 'dump.sqlite')
        local_context = contextlib.ExitStack()
        with local_context:
            try:
                dump_file = local_context.enter_context(storage.open(
                    dump_filename, 'rb', on_filesystem=True, noop=scope['mpi_rank']))
            except FileNotFoundError:
                return -1, {}
            traj = local_context.enter_context(gtar.GTAR(dump_file.name, 'r'))

            # grab per-type distance arrays
            for (frame, trans) in traj.recordsNamed('type_translation_distance'):
                if int(frame) <= scope['cumulative_steps']:
                    distances['translation'] = trans
                    frame = int(frame)
                else:
                    break

            for (frame, rot) in traj.recordsNamed('type_rotation_distance'):
                if int(frame) <= scope['cumulative_steps']:
                    distances['rotation'] = rot
                    frame = int(frame)
                else:
                    break

        if distances:
            types = context.snapshot.particles.types
            kwargs = dict()

            if 'translation' in distances:
                kwargs['d'] = dict(zip(types, distances['translation']))

            if 'rotation' in distances:
                kwargs['a'] = dict(zip(types, distances['rotation']))

            integrator.set_params(**kwargs)

        msg = 'Loaded tuned move distances from {} for frame {}: {}'.format(
            dump_filename, frame, distances)
        logger.debug(msg)

        return frame, distances

class TuneHPMC(RunHPMC):
    """Helper class to be inserted as a stage immediately before a RunHPMC stage.

    Sets up a move size tuner, runs a certain number of steps while
    updating, and then discards the system state while saving the move
    size state.
    """

    ARGS = RunHPMC.ARGS + [
        Arg('tune_acceptance_ratio', None, float),
        Arg('tune_epochs', None, int),
    ]

    def setup_dumps(self, scope, storage, context):
        # disable dumps for this pseudo-stage
        pass

    def run_steps(self, scope, storage, context):
        context.cancel_saving()

        epochs = self.arguments['tune_epochs']
        acceptance_ratio = self.arguments['tune_acceptance_ratio']
        steps_per_epoch = self.arguments['steps']/epochs

        tunables = ['d'] if 'type_shapes' not in scope else ['d', 'a']
        integrator = scope['integrator']
        tuner = hoomd.hpmc.util.tune(
            integrator, tunables, target=acceptance_ratio)

        for epoch in range(epochs):
            hoomd.run(steps_per_epoch)
            tuner.update()

        self.save_tune_results(scope, storage, context)

    def save_tune_results(self, scope, storage, context):
        import gtar

        # step count before this stage
        beginning_steps = scope['cumulative_steps'] - self.arguments['steps']
        types = context.snapshot.particles.types

        if scope['mpi_rank']:
            return

        integrator = scope['integrator']
        translation, rotation = [], []

        for t in types:
            translation.append(integrator.get_d(t))
            rotation.append(integrator.get_a(t))

        dump_filename = scope.get('dump_filename', 'dump.sqlite')

        msg = 'Dumping tuned move distances to {}: translation {}, rotation {}'.format(
            dump_filename, translation, rotation)
        logger.debug(msg)

        local_context = contextlib.ExitStack()
        with local_context:
            dump_file = local_context.enter_context(storage.open(
                dump_filename, 'ab', on_filesystem=True, noop=scope['mpi_rank']))
            getar_file = local_context.enter_context(gtar.GTAR(dump_file.name, 'a'))

            path = 'hpmc/frames/{}/type_translation_distance.f32.uni'.format(beginning_steps)
            getar_file.writePath(path, translation)
            path = 'hpmc/frames/{}/type_rotation_distance.f32.uni'.format(beginning_steps)
            getar_file.writePath(path, rotation)

def is_concave(polygon):
    vertices = np.array(polygon['vertices'], dtype=np.float32)

    delta = np.roll(vertices, -1, axis=0) - vertices
    return np.any(np.cross(delta, np.roll(delta, -1, axis=0)) < 0)
