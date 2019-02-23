import collections
import itertools
import logging

import numpy as np

from .internal import intfloat, HoomdContext
import hoomd
import hoomd.md
import flowws

logger = logging.getLogger(__name__)

class Run(flowws.Stage):
    ARGS = list(itertools.starmap(
        flowws.Stage.ArgumentSpecification,
        [
            ('steps', intfloat, None, 'Number of timesteps to run'),
            ('timestep_size', float, .005, 'Timestep size'),
            ('integrator', str, None, 'Integrator type'),
            ('temperature', float, 1, 'Temperature for isothermal simulations'),
            ('tau_t', float, 1, 'Thermostat time constant for isothermal simulations'),
            ('pressure', float, 1, 'Pressure for isobaric simulations'),
            ('tau_p', float, 10, 'Barostat time constant for isobaric simulations'),
            ('bd_seed', int, 12, 'Random number seed for Brownian/Langevin thermostats'),
            ('backup_period', intfloat, 0, 'Period for dumping a backup file'),
            ('dump_period', intfloat, 0, 'Period for dumping a trajectory file'),
            ('expand_by', float, None, 'Expand each dimension of the box by this ratio during this stage'),
            ('compress_to', float, None, 'Compress to the given packing fraction during this stage (overrides expand_by)'),
        ]
    ))

    def setup_integrator(self, scope, storage):
        integrator_type = self.arguments['integrator']

        if integrator_type == 'nve':
            integrator = hoomd.md.integrate.nve(hoomd.group.all())
        elif integrator_type == 'nvt':
            kT = self.arguments['temperature']
            tau = self.arguments['tau_t']
            integrator = hoomd.md.integrate.nvt(
                hoomd.group.all(), kT=kT, tau=tau)
        elif integrator_type == 'langevin':
            kT = self.arguments['temperature']
            seed = self.arguments['bd_seed']
            integrator = hoomd.md.integrate.langevin(
                hoomd.group.all(), kT=kT, seed=seed)
        elif integrator_type == 'brownian':
            kT = self.arguments['temperature']
            seed = self.arguments['bd_seed']
            integrator = hoomd.md.integrate.brownian(
                hoomd.group.all(), kT=kT, seed=seed)
        elif integrator_type == 'npt':
            kT = self.arguments['temperature']
            tau = self.arguments['tau_t']
            pressure = self.arguments['pressure']
            tauP = self.arguments['tau_p']
            integrator = hoomd.md.integrate.npt(
                hoomd.group.all(), kT=kT, tau=tau, P=pressure, tauP=tauP)
        else:
            raise NotImplementedError(
                'Unknown integrator type {}'.format(integrator_type))

        hoomd.md.integrate.mode_standard(dt=self.arguments['timestep_size'])

        return integrator

    def setup_dumps(self, scope, storage, context):
        if self.arguments['backup_period']:
            backup_filename = scope.get('restore_filename', 'backup.tar')
            backup_file = context.enter_context(
                storage.open(backup_filename, 'wb', on_filesystem=True))
            hoomd.dump.getar.simple(
                backup_file.name,  self.arguments['backup_period'], '1',
                static=[], dynamic=['all'])

        if self.arguments['dump_period']:
            dump_filename = scope.get('dump_filename', 'dump.sqlite')
            dump_file = context.enter_context(
                storage.open(dump_filename, 'ab', on_filesystem=True))

            dynamic_quantities = (
                ['viz_aniso_dynamic'] if 'type_shapes' in scope else ['viz_dynamic'])

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

    def run(self, scope, storage):
        callbacks = scope.setdefault('callbacks', collections.defaultdict(list))
        scope['cumulative_steps'] = (scope.get('cumulative_steps', 0) +
                                     self.arguments['steps'])

        with HoomdContext(scope, storage) as ctx:
            if ctx.check_timesteps():
                return

            self.setup_integrator(scope, storage)

            self.setup_dumps(scope, storage, ctx)

            self.setup_compression(scope, storage, ctx)

            for c in callbacks['pre_run']:
                c(scope, storage)

            hoomd.run_upto(scope['cumulative_steps'])

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
