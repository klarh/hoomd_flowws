import argparse
import collections
import itertools

import hoomd
import flowws
from flowws import Argument as Arg

class Interaction(flowws.Stage):
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

    def run(self, scope, storage):
        callbacks = scope.setdefault('callbacks', collections.defaultdict(list))

        if self.arguments['reset']:
            pre_run_callbacks = [c for c in callbacks['pre_run']
                                 if not isinstance(c, Interaction)]
            callbacks['pre_run'] = pre_run_callbacks

        callbacks['pre_run'].append(self)

    def __call__(self, scope, storage):
        interaction_type = self.arguments['type']
        params = dict(self.arguments['params'])

        nlist = hoomd.md.nlist.tree()
        system = scope['system']

        if interaction_type == 'pair.lj':
            kwargs = dict(self.arguments['global_params'])
            kwargs['nlist'] = nlist
            interaction = hoomd.md.pair.lj(**kwargs)

            pair_params = collections.defaultdict(dict)

            for (a, b), name, value in self.arguments['pair_params']:
                if a == '_':
                    a = system.particles.types
                if b == '_':
                    b = system.particles.types
                interaction.pair_coeff.set(a, b, **{name: value})

            interaction.set_params(mode='shift')
        else:
            raise NotImplementedError(
                'Unknown Interaction type {}'.format(interaction_type))
