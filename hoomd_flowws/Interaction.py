import argparse
import collections
import itertools

import hoomd
import flowws

class Interaction(flowws.Stage):
    ARGS = list(itertools.starmap(
        flowws.Stage.ArgumentSpecification,
        [
            ('reset', bool, False, 'Clear previously-defined interactions beforehand'),
            ('type', str, None, 'Interaction class name'),
            ('params', eval, None, 'Interaction parameters'),
        ]
    ))

    def run(self, scope, storage):
        callbacks = scope.setdefault('callbacks', collections.defaultdict(list))

        if self.arguments['reset']:
            pre_run_callbacks = [c for c in callbacks['pre_run']
                                 if not isinstance(c, Interaction)]
            callbacks['pre_run'] = pre_run_callbacks

        callbacks['pre_run'].append(self)

    @classmethod
    def from_command(cls, args):
        parser = argparse.ArgumentParser(
            prog=cls.__name__, description=cls.__doc__)

        parser.add_argument('--reset', default=False, action='store_true',
            help=cls.ARGS[0].description)
        parser.add_argument('--type', required=True,
            help=cls.ARGS[1].description)
        parser.add_argument('-g', '--global-params', nargs=2, action='append', default=[],
            metavar=('name', 'value'),
            help='Set global parameters of the interaction')
        parser.add_argument('-p', '--pair-params', nargs=4, action='append', default=[],
            metavar=('A', 'B', 'name', 'value'),
            help='Set type-pair interaction parameters (_ for all types)')

        args = parser.parse_args(args)

        pair_params = collections.defaultdict(dict)
        for (a, b, name, value) in args.pair_params:
            pair_params[(a, b)][name] = eval(value)

        pair_params_list = []
        for (a, b, _, _) in args.pair_params:
            if (a, b) in pair_params:
                element = (a, b, pair_params.pop((a, b)))
                pair_params_list.append(element)

        potential_params = dict(pair_params=pair_params_list)

        for (name, value) in args.global_params:
            potential_params[name] = eval(value)

        return cls(reset=args.reset, type=args.type, params=potential_params)

    def __call__(self, scope, storage):
        interaction_type = self.arguments['type']
        params = dict(self.arguments['params'])

        nlist = hoomd.md.nlist.tree()
        system = scope['system']

        if interaction_type == 'pair.lj':
            pair_params = params.pop('pair_params')
            kwargs = dict(nlist=nlist, **params)
            interaction = hoomd.md.pair.lj(**kwargs)

            for (a, b, kwargs) in pair_params:
                if a == '_':
                    a = system.particles.types
                if b == '_':
                    b = system.particles.types
                interaction.pair_coeff.set(a, b, **kwargs)

            interaction.set_params(mode='shift')
        else:
            raise NotImplementedError(
                'Unknown Interaction type {}'.format(interaction_type))
