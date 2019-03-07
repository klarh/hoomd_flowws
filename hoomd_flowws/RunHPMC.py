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

BOX_MOVE_NAMES = ['aspect', 'length', 'shear', 'volume', 'ln_volume']

@flowws.add_stage_arguments
class RunHPMC(Run):
    """Run for some number of steps using HPMC"""
    ARGS = [
        Arg('steps', '-s', intfloat, None, required=True,
            help='Number of timesteps to run'),
        Arg('integrator', '-i', str, None, required=True,
            help='Integrator type'),
        Arg('pressure', None, float, 1,
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
            'number of steps at each epoch'),
        Arg('box_move_aspect', None, (float, float), metavar=('distance', 'weight'),
            help='Move distance and weight for box aspect ratio moves'),
        Arg('box_move_length', None, (float, float), metavar=('distance', 'weight'),
            help='Move distance and weight for box length moves'),
        Arg('box_move_ln_volume', None, (float, float), metavar=('distance', 'weight'),
            help='Move distance and weight for box log-volume moves'),
        Arg('box_move_shear', None, (float, float), metavar=('distance', 'weight'),
            help='Move distance and weight for box shear moves'),
        Arg('box_move_volume', None, (float, float), metavar=('distance', 'weight'),
            help='Move distance and weight for box volume moves'),
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
            if any('box_move_{}'.format(name) in self.arguments
                   for name in ['length', 'volume', 'ln_volume']):
                raise ArgumentError('NVT integration mode specified, but box moves were also enabled')
        elif integrator_type == 'npt':
            if all('box_move_{}'.format(name) not in self.arguments
                   for name in ['length', 'volume', 'ln_volume']):
                raise ArgumentError('Must enable box moves for NPT simulations')
        else:
            raise NotImplementedError(integrator_type)

        updater = self.setup_box_updater(scope, storage, context, integrator)

        frame, _ = self.load_move_distance(scope, storage, context, integrator, updater)
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
            self.load_move_distance(scope, storage, context, integrator, updater)

        scope['integrator'] = integrator

        return integrator

    def setup_box_updater(self, scope, storage, context, integrator):
        should_create = any(self.arguments.get('box_move_{}'.format(name), False)
                            for name in BOX_MOVE_NAMES)
        should_create |= self.arguments['integrator'] == 'npt'

        updater = None
        if should_create:
            updater = hoomd.hpmc.update.boxmc(
                integrator, self.arguments['pressure'],
                self.arguments['integrator_seed'] + 20)

            for name in BOX_MOVE_NAMES:
                if 'box_move_{}'.format(name) not in self.arguments:
                    continue
                (distance, weight) = self.arguments['box_move_{}'.format(name)]
                getattr(updater, name)(delta=distance, weight=weight)

            scope['box_updater'] = updater

        return updater

    def load_move_distance(self, scope, storage, context, integrator, updater):
        import gtar

        # per-type move distance arrays for translation/rotation/box
        distances = {}
        frame = -1

        dump_filename = scope.get('dump_filename', 'dump.sqlite')
        local_context = contextlib.ExitStack()
        with local_context:
            try:
                dump_file = local_context.enter_context(storage.open(
                    dump_filename, 'rb', on_filesystem=True, noop=scope['mpi_rank']))
                traj = local_context.enter_context(gtar.GTAR(dump_file.name, 'r'))
            except (FileNotFoundError, RuntimeError):
                return -1, {}

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

            for name in BOX_MOVE_NAMES:
                box_name = 'box_{}'.format(name)
                distance_name = '{}_distance'.format(box_name)
                for (frame, rot) in traj.recordsNamed(distance_name):
                    if int(frame) <= scope['cumulative_steps']:
                        distances[box_name] = rot
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

            for (key, distance) in distances.items():
                if key.startswith('box_') and updater is not None:
                    name = key[4:]
                    getattr(updater, name)(delta=distance)

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

        box_tuner = None
        updater = scope.get('box_updater', None)
        if updater is not None:
            tunable_map = dict(
                length=['dLx', 'dLy', 'dLz'],
                volume=['dV'],
                ln_volume=['dlnV'],
                shear=['dxy', 'dxz', 'dyz'],
                aspect=[] # aspect tuning is not builtin
            )
            tunables = sum(
                (tunable_map[name] for name in BOX_MOVE_NAMES if
                 'box_move_{}'.format(name) in self.arguments), [])
            box_tuner = hoomd.hpmc.util.tune_npt(
                updater, tunables, target=acceptance_ratio)

        for epoch in range(epochs):
            hoomd.run(steps_per_epoch)
            tuner.update()

            if box_tuner is not None:
                box_tuner.update()

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

        updater = scope.get('box_updater', None)
        box_distances = {}
        if updater is not None:
            for name in BOX_MOVE_NAMES:
                box_distances[name] = getattr(updater, name)()['delta']

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

            for (key, value) in box_distances.items():
                path = 'hpmc/frames/{}/box_{}_distance.f32.uni'.format(
                    beginning_steps, key)
                getar_file.writePath(path, value)

def is_concave(polygon):
    vertices = np.array(polygon['vertices'], dtype=np.float32)

    delta = np.roll(vertices, -1, axis=0) - vertices
    return np.any(np.cross(delta, np.roll(delta, -1, axis=0)) < 0)
