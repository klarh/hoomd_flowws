import argparse
import collections
import itertools

import hoomd, hoomd.md
import flowws
from flowws import Argument as Arg

@flowws.add_stage_arguments
class Interaction(flowws.Stage):
    """Specify a new interaction potential to include in future MD stages"""
    ARGS = [
        Arg('reset', '-r', bool, False,
            help='Clear previously-defined interactions beforehand'),
        Arg('type', '-t', str, required=True,
            help='Interaction class name'),
        Arg('global_params', '-g', [(str, eval)],
            metavar=('name', 'value'),
            help='Global parameters of the interaction'),
        Arg('pair_params', '-p', [(str, str, str, eval)],
            metavar=('A', 'B', 'name', 'value'),
            help='Type pair-based parameters of the interaction')
    ]

    GLOBAL_INTERACTION_MAP = {}

    def run(self, scope, storage):
        """Registers this object to provide a force compute in future MD stages"""
        callbacks = scope.setdefault('callbacks', collections.defaultdict(list))

        if self.arguments['reset']:
            pre_run_callbacks = [c for c in callbacks['pre_run']
                                 if not isinstance(c, Interaction)]
            callbacks['pre_run'] = pre_run_callbacks

        callbacks['pre_run'].append(self)

    def __call__(self, scope, storage, context):
        """Callback to be performed before each run command.

        Initializes a pair potential interaction based on per-type
        shape information.
        """
        interaction_type = self.arguments['type']

        nlist = hoomd.md.nlist.tree()
        system = scope['system']

        try:
            interaction_type = self.GLOBAL_INTERACTION_MAP[interaction_type]
        except KeyError:
            raise NotImplementedError(
                'Unknown Interaction type {}'.format(interaction_type))

        kwargs = dict(self.arguments['global_params'])
        kwargs['nlist'] = nlist
        interaction = interaction_type(**kwargs)

        pair_params = collections.defaultdict(dict)

        for a, b, name, value in self.arguments['pair_params']:
            if a == '_':
                a = system.particles.types
            if b == '_':
                b = system.particles.types
            interaction.pair_coeff.set(a, b, **{name: value})

        interaction.set_params(mode='shift')

    @classmethod
    def register_interaction(cls, interaction, *names):
        """Convenience method to bind one or more names to a force-generation function"""
        for name in names:
            cls.GLOBAL_INTERACTION_MAP[name] = interaction

Interaction.register_interaction(hoomd.md.pair.buckingham, 'pair.buckingham', 'buckingham')
Interaction.register_interaction(hoomd.md.pair.dipole, 'pair.dipole', 'dipole')
Interaction.register_interaction(hoomd.md.pair.dpd, 'pair.dpd', 'dpd')
Interaction.register_interaction(hoomd.md.pair.dpdlj, 'pair.dpdlj', 'dpdlj')
Interaction.register_interaction(hoomd.md.pair.dpd_conservative, 'pair.dpd_conservative', 'dpd_conservative')
Interaction.register_interaction(hoomd.md.pair.ewald, 'pair.ewald', 'ewald')
Interaction.register_interaction(hoomd.md.pair.force_shifted_lj, 'pair.force_shifted_lj', 'force_shifted_lj')
Interaction.register_interaction(hoomd.md.pair.gauss, 'pair.gauss', 'gauss')
Interaction.register_interaction(hoomd.md.pair.gb, 'pair.gb', 'gb')
Interaction.register_interaction(hoomd.md.pair.lj, 'pair.lj', 'lj')
Interaction.register_interaction(hoomd.md.pair.lj1208, 'pair.lj1208', 'lj1208')
Interaction.register_interaction(hoomd.md.pair.mie, 'pair.mie', 'mie')
Interaction.register_interaction(hoomd.md.pair.morse, 'pair.morse', 'morse')
Interaction.register_interaction(hoomd.md.pair.moliere, 'pair.moliere', 'moliere')
Interaction.register_interaction(hoomd.md.pair.reaction_field, 'pair.reaction_field', 'reaction_field')
Interaction.register_interaction(hoomd.md.pair.slj, 'pair.slj', 'slj')
Interaction.register_interaction(hoomd.md.pair.square_density, 'pair.square_density', 'square_density')
Interaction.register_interaction(hoomd.md.pair.tersoff, 'pair.tersoff', 'tersoff')
Interaction.register_interaction(hoomd.md.pair.yukawa, 'pair.yukawa', 'yukawa')
Interaction.register_interaction(hoomd.md.pair.zbl, 'pair.zbl', 'zbl')
