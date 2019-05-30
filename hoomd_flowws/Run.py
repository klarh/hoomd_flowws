import collections
import itertools
import logging
import os

import numpy as np

from .internal import intfloat, HoomdContext
import hoomd
import hoomd.md
import flowws
from flowws import Argument as Arg

logger = logging.getLogger(__name__)

@flowws.add_stage_arguments
class Run(flowws.Stage):
    """Run for a given number of timesteps using MD"""
    ARGS = [
        Arg('steps', '-s', intfloat, None,
            help='Number of timesteps to run'),
        Arg('timestep_size', None, float, 0.005,
            help='Timestep size'),
        Arg('integrator', None, str, required=True,
            help='Integrator type'),
        Arg('temperature', None, float, 1,
            help='Temperature for isothermal simulations'),
        Arg('ramp_t_to', None, float, None,
            help='Ramp temperature to the given value over time during this stage'),
        Arg('tau_t', None, float, 1,
            help='Thermostat time constant for isothermal simulations'),
        Arg('pressure', None, float, 1,
            help='Pressure for isobaric simulations'),
        Arg('tau_p', None, float, 10,
            help='Barostat time constant for isobaric simulations'),
        Arg('bd_seed', None, int, 12,
            help='Random number seed for Brownian/Langevin thermostats'),
        Arg('zero_momentum', '-z', intfloat, 0,
            help='Period for zeroing the momentum of the system (default: disabled)'),
        Arg('backup_period', '-b', intfloat, 0,
            help='Period for dumping a backup file'),
        Arg('dump_period', '-d', intfloat, 0,
            help='Period for dumping a trajectory file'),
        Arg('trajectory_quantities', None, [str], [],
            help='Additional trajectory quantities to dump'),
        Arg('dump_quantities', '-q', (intfloat, str),
            help='Dump period and comma-separated list of quantities to record (i.e. temperature,pressure)'),
        Arg('expand_by', None, float,
            help='Expand each dimension of the box by this ratio during this stage'),
        Arg('compress_to', None, float,
            help='Compress to the given packing fraction during this stage (overrides expand_by)'),
    ]

    def setup_integrator(self, scope, storage, context):
        integrator_type = self.arguments['integrator']

        kT = self.arguments['temperature']

        if self.arguments.get('ramp_t_to', None) is not None:
            times = [0, self.arguments['steps']]
            temperatures = [self.arguments['temperature'], self.arguments['ramp_t_to']]
            kT = hoomd.variant.linear_interp(
                list(zip(times, temperatures)), zero=scope['previous_steps'])

        if integrator_type == 'nve':
            integrator = hoomd.md.integrate.nve(hoomd.group.all())
        elif integrator_type == 'nvt':
            tau = self.arguments['tau_t']
            integrator = hoomd.md.integrate.nvt(
                hoomd.group.all(), kT=kT, tau=tau)
        elif integrator_type == 'langevin':
            seed = self.arguments['bd_seed']
            integrator = hoomd.md.integrate.langevin(
                hoomd.group.all(), kT=kT, seed=seed)
        elif integrator_type == 'brownian':
            seed = self.arguments['bd_seed']
            integrator = hoomd.md.integrate.brownian(
                hoomd.group.all(), kT=kT, seed=seed)
        elif integrator_type == 'npt':
            tau = self.arguments['tau_t']
            pressure = self.arguments['pressure']
            tauP = self.arguments['tau_p']
            integrator = hoomd.md.integrate.npt(
                hoomd.group.all(), kT=kT, tau=tau, P=pressure, tauP=tauP)
        else:
            raise NotImplementedError(
                'Unknown integrator type {}'.format(integrator_type))

        hoomd.md.integrate.mode_standard(dt=self.arguments['timestep_size'])

        if self.arguments['zero_momentum']:
            hoomd.md.update.zero_momentum(self.arguments['zero_momentum'])

        return integrator

    def setup_dumps(self, scope, storage, context):
        if self.arguments['backup_period']:
            backup_filename = context.get_backup_filename()
            backup_file = context.enter_context(
                storage.open(backup_filename, 'wb', on_filesystem=True,
                             noop=scope['mpi_rank']))
            hoomd.dump.getar.simple(
                backup_file.name,  self.arguments['backup_period'], '1',
                static=[], dynamic=['all'])

        if self.arguments.get('dump_quantities', None):
            dump_filename = scope.get('dump_filename', 'dump.sqlite')
            dump_filename = '.'.join(
                (os.path.splitext(dump_filename)[0], 'quantities', 'log'))
            (period, quantities) = self.arguments['dump_quantities']
            quantities = [q.strip() for q in quantities.split(',')]

            log_file = context.enter_context(
                storage.open(dump_filename, 'ab', on_filesystem=True,
                             noop=scope['mpi_rank']))

            hoomd.analyze.log(log_file.name, quantities, period)

        if self.arguments['dump_period']:
            dump_filename = scope.get('dump_filename', 'dump.sqlite')
            dump_file = context.enter_context(
                storage.open(dump_filename, 'ab', on_filesystem=True,
                             noop=scope['mpi_rank']))

            dynamic_quantities = (
                ['viz_aniso_dynamic'] if 'type_shapes' in scope else ['viz_dynamic'])

            dynamic_quantities.extend(self.arguments.get('trajectory_quantities', []))

            dump = hoomd.dump.getar.simple(
                dump_file.name,  self.arguments['dump_period'], 'a',
                static=['viz_static'], dynamic=dynamic_quantities)

            if 'type_shapes' in scope:
                type_shapes = scope['type_shapes']
                dump.writeJSON('type_shapes.json', type_shapes, True)

    def setup_compression(self, scope, storage, context):
        if not self.arguments.get('compress_to', None) and not self.arguments.get('expand_by', None):
            return

        dimensions = scope.get('dimensions', 3)

        if self.arguments.get('compress_to', None):
            current_phi = compute_packing_fraction(scope, storage, context.snapshot)
            volume_ratio = current_phi/self.arguments['compress_to']
            length_ratio = volume_ratio**(1./dimensions)
            self.arguments['expand_by'] = length_ratio

        box = context.snapshot.box

        factor = self.arguments['expand_by']
        times = [0, self.arguments['steps']]
        Lx = hoomd.variant.linear_interp(
            list(zip(times, [box.Lx, box.Lx*factor])), zero='now')
        Ly = hoomd.variant.linear_interp(
            list(zip(times, [box.Ly, box.Ly*factor])), zero='now')
        if dimensions == 2:
            Lz = box.Lz
        else:
            Lz = hoomd.variant.linear_interp(
                list(zip(times, [box.Lz, box.Lz*factor])), zero='now')

        updater = hoomd.update.box_resize(Lx=Lx, Ly=Ly, Lz=Lz)
        return updater

    def run_steps(self, scope, storage, context):
        hoomd.run_upto(scope['cumulative_steps'])

    def run(self, scope, storage):
        callbacks = scope.setdefault('callbacks', collections.defaultdict(list))
        scope['previous_steps'] = scope.get('cumulative_steps', 0)
        scope['cumulative_steps'] = (scope['previous_steps'] +
                                     self.arguments['steps'])

        with HoomdContext(scope, storage) as ctx:
            if ctx.check_timesteps():
                return

            self.setup_integrator(scope, storage, ctx)

            self.setup_dumps(scope, storage, ctx)

            self.setup_compression(scope, storage, ctx)

            for c in callbacks['pre_run']:
                c(scope, storage, ctx)

            self.run_steps(scope, storage, ctx)

            for c in callbacks['post_run']:
                c(scope, storage, ctx)

def compute_packing_fraction(scope, storage, snapshot):
    dimensions = scope.get('dimensions', 3)

    type_volumes = []
    for shape in scope.get('type_shapes', []):
        volume = np.polyval(shape['rounding_volume_polynomial'], shape['rounding_radius'])
        type_volumes.append(volume)

    if not type_volumes:
        logger.warning('No shape information found, assuming particles are spheres with diameter 1')
        if dimensions == 2:
            type_volumes = len(snapshot.particles.types)*[np.pi*0.25]
        else:
            type_volumes = len(snapshot.particles.types)*[4./3*np.pi*0.125]

    type_volumes = np.array(type_volumes, dtype=np.float32)

    particle_volume = np.sum(type_volumes[snapshot.particles.typeid])

    box_volume = snapshot.box.get_volume()

    return particle_volume/box_volume
