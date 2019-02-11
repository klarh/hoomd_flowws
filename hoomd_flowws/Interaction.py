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

    def __call__(self, scope, storage):
        interaction_type = self.arguments['type']
        params = self.arguments['params']

        if 'nlist' not in scope:
            scope['nlist'] = hoomd.md.nlist.tree()
        nlist = scope['nlist']

        if interaction_type == 'pair.lj':
            interaction = hoomd.md.pair.lj(params['rcut'], nlist)

            for (a, b, kwargs) in params['pair_params']:
                interaction.pair_coeff.set(a, b, **kwargs)

            interaction.set_params(mode='shift')
        else:
            raise NotImplementedError(
                'Unknown Interaction type {}'.format(interaction_type))
