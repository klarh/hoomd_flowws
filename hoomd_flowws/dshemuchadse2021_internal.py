
import flowws
import hoomd
import numpy as np

class Dshemuchadse2021InteractionBase(flowws.Stage):
    def find_potential_parameters(self):
        test_rs = np.linspace(
            self.arguments['r_min'], 8*self.arguments['r_min'], self.arguments['width'])
        test_U = self.potential(test_rs, **self.potential_kwargs)
        test_F = self.force(test_rs, **self.potential_kwargs)
        dUdr = -test_F

        try:
            # step 1: find first maximum (dU/dR = 0) after second attractive (U < 0) minimum
            minima_locations = np.all([dUdr[1:] >= 0, dUdr[:-1] < 0, test_U[:-1] < 0], axis=0)
            minima_indices = np.cumsum(minima_locations)

            # domain of the function between the second and third minima
            second_minimum_indices = np.where(minima_indices == 2)[0]
            search_min = test_rs[second_minimum_indices[0]]
            search_max = test_rs[second_minimum_indices[-1]]

            # bisect to find dU/dR = 0
            left, right = search_min, search_max
            while right - left > 1e-7:
                center = .5*(left + right)
                fcenter = -self.force(center, **self.potential_kwargs)
                if fcenter >= 0:
                    left = center
                else:
                    right = center
            rmax = .5*(left + right)
        except IndexError: # LJG doesn't necessarily have enough minima
            rmax = None

        # step 2: find minimum U value of the function to rescale to -1
        imin = np.argmin(test_U)
        left, right = test_rs[imin - 1], test_rs[imin + 1]
        while right - left > 1e-7:
            center = .5*(left + right)
            fcenter = -self.force(center, **self.potential_kwargs)
            if fcenter < 0:
                left = center
            else:
                right = center
        center = .5*(left + right)
        Umin = self.potential(center, **self.potential_kwargs)

        return rmax, 1./np.abs(Umin)

    def __call__(self, scope, storage, context):
        """Callback to be performed before each run command.

        Initializes a pair potential interaction for all pairs of types.
        """
        nlist = hoomd.md.nlist.tree()
        system = scope['system']

        def table_callable(r, rmin, rmax, **kwargs):
            return self.potential(r, **kwargs), self.force(r, **kwargs)

        rmin = self.arguments['r_min']
        rmax = self.rmax
        width = self.arguments['width']

        table = hoomd.md.pair.table(width=width, nlist=nlist)
        all_types = list(system.particles.types)
        table.pair_coeff.set(
            all_types, all_types,
            func=table_callable, rmin=rmin, rmax=rmax, coeff=self.potential_kwargs)

    def draw_matplotlib(self, figure):
        ax = figure.add_subplot(111)
        test_rs = np.linspace(self.arguments['r_min'], self.rmax, self.arguments['width'])
        test_Us = self.potential(test_rs, **self.potential_kwargs)
        ax.plot(test_rs, test_Us)
        ax.set_ylim(-1, 5)
        ax.set_xlabel('r')
        ax.set_ylabel('Potential')
