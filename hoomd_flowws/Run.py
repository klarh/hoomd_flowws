import collections
import itertools

from .internal import HoomdContext
import hoomd
import hoomd.md
import flowws

class Run(flowws.Stage):
    ARGS = list(itertools.starmap(
        flowws.Stage.ArgumentSpecification,
        [
            ('steps', int, None, 'Number of timesteps to run'),
            ('timestep_size', float, .005, 'Timestep size'),
            ('integrator', str, None, 'Integrator type'),
            ('integrator_params', eval, {}, 'Parameters for integrator'),
            ('backup_period', int, 0, 'Period for dumping a backup file'),
            ('dump_period', int, 0, 'Period for dumping a trajectory file'),
        ]
    ))

    def setup_integrator(self, scope, storage):
        integrator_type = self.arguments['integrator']
        integrator_params = self.arguments['integrator_params']

        if integrator_type == 'nve':
            integrator = hoomd.md.integrate.nve(hoomd.group.all())
        elif integrator_type == 'nvt':
            integrator = hoomd.md.integrate.nvt(
                hoomd.group.all(), **integrator_params)
        elif integrator_type == 'langevin':
            integrator = hoomd.md.integrate.langevin(
                hoomd.group.all(), **integrator_params)
        elif integrator_type == 'npt':
            integrator = hoomd.md.integrate.npt(
                hoomd.group.all(), **integrator_params)
        else:
            raise NotImplementedError(
                'Unknown integrator type {}'.format(integrator_type))

        hoomd.md.integrate.mode_standard(dt=self.arguments['timestep_size'])

        return integrator

    def run(self, scope, storage):
        callbacks = scope.setdefault('callbacks', collections.defaultdict(list))
        scope['cumulative_steps'] = (scope.get('cumulative_steps', 0) +
                                     self.arguments['steps'])

        with HoomdContext(scope, storage) as ctx:
            if ctx.check_timesteps():
                return

            self.setup_integrator(scope, storage)

            if self.arguments['backup_period']:
                backup_filename = scope.get('restore_filename', 'backup.tar')
                backup_file = ctx.enter_context(
                    storage.open(backup_filename, 'wb', on_filesystem=True))
                hoomd.dump.getar.simple(
                    backup_file.name,  self.arguments['backup_period'], '1',
                    static=[], dynamic=['all'])

            if self.arguments['dump_period']:
                dump_filename = scope.get('dump_filename', 'dump.sqlite')
                dump_file = ctx.enter_context(
                    storage.open(dump_filename, 'wb', on_filesystem=True))

                dump = hoomd.dump.getar.simple(
                    dump_file.name,  self.arguments['dump_period'], 'a',
                    static=['viz_static'], dynamic=['viz_aniso_dynamic'])

                if 'type_shapes' in scope:
                    type_shapes = scope['type_shapes']
                    print(type_shapes)
                    dump.writeJSON('type_shapes.json', type_shapes, True)

            for c in callbacks['pre_run']:
                c(scope, storage)

            hoomd.run_upto(scope['cumulative_steps'])
