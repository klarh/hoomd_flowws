import argparse
import collections
import itertools
import re

import numpy as np
import flowws

import hoomd.dem

# Simple container for shape-generation functions to return
ShapeInfo = collections.namedtuple(
    'ShapeInfo', ['vertices', 'rounding_volume_polynomial'])

class ShapeDefinition(flowws.Stage):
    ARGS = list(itertools.starmap(
        flowws.Stage.ArgumentSpecification,
        [
            ('shape_parameters', eval, None, 'Per-type shape definitions and parameters'),
        ]
    ))

    def run(self, scope, storage):
        shape_parameters = self.arguments['shape_parameters']
        type_shapes = [make_shape(params) for params in shape_parameters]

        if type_shapes and len(type_shapes[0]['vertices'][0]) == 2:
            scope['dimensions'] = 2

        scope['type_shapes'] = type_shapes

    @classmethod
    def from_command(cls, args):
        parser = argparse.ArgumentParser(
            prog=cls.__name__, description=cls.__doc__)

        parser.add_argument('-a', '--shape-arg', action=StoreShapeParams, nargs='*',
                            default=collections.defaultdict(list),
                            help='Add a shape argument to the list (ex. -a num_vertices 6 -a scale 2) for '
                            'a previously-specified shape in the argument list. To '
                            'specify a new shape, use -a shape <shapeName>.')

        args = parser.parse_args(args)

        shapes = []
        for type_index in list(sorted(args.shape_arg)):
            modifications = []
            shape = dict(modifications=modifications)
            for (key, val) in args.shape_arg[type_index]:
                if key == 'shape':
                    shape['type'] = val
                elif key == 'scale':
                    modifications.append(dict(type=key, factor=val[0]))
                elif key == 'round':
                    modifications.append(dict(type=key, radius=val[0]))
                elif key == 'unit_volume':
                    modifications.append(dict(type=key))
                else:
                    shape[key] = val[0]

            shapes.append(shape)

        return cls(shape_parameters=shapes)

class StoreShapeParams(argparse.Action):
    """argparse action to store shape parameters to allow for multiple
    shapes in the command line"""
    def __init__(self, *args, **kwargs):
        self._curType = -1
        super(StoreShapeParams, self).__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if values[0] == 'shape':
            self._curType += 1
            namespace.shape_arg[self._curType].append(tuple(values))
        else:
            rest = [eval(v) for v in values[1:]]
            namespace.shape_arg[self._curType].append((values[0], rest))

class Shape:
    """Helper class to make shape definitions more succinct"""
    convex_polyhedron_functions = {}
    polygon_functions = {}

    @classmethod
    def convex_polyhedron(cls, function):
        """Decorator to register a convex polyhedron generator function"""
        name = function.__name__

        pattern = re.compile(r'^(?P<name>[a-zA-Z_]+)_shape$')
        match = pattern.match(name)

        assert match, 'convex_polyhedron-wrapped functions must be named <name>_shape'

        shape_type = match.group('name')

        cls.convex_polyhedron_functions[shape_type] = function
        return function

    @classmethod
    def convex_polyhedron_shapedef(cls, name, params):
        """Get a previously-registered convex polyhedron by its name"""
        shape_info = cls.convex_polyhedron_functions[name](**params)
        vertices = np.array(shape_info.vertices, dtype=np.float32).tolist()

        result = dict(
            type='ConvexPolyhedron', vertices=vertices, rounding_radius=0)

        rmax = np.max(np.linalg.norm(vertices, axis=-1))
        result['circumsphere_radius'] = rmax
        result['rounding_volume_polynomial'] = shape_info.rounding_volume_polynomial

        return result

    @classmethod
    def polygon(cls, function):
        """Decorator to register a convex polyhedron generator function"""
        name = function.__name__

        pattern = re.compile(r'^(?P<name>[a-zA-Z_]+)_shape$')
        match = pattern.match(name)

        assert match, 'polygon-wrapped functions must be named <name>_shape'

        shape_type = match.group('name')

        cls.polygon_functions[shape_type] = function
        return function

    @classmethod
    def polygon_shapedef(cls, name, params):
        """Get a previously-registered convex polyhedron by its name"""
        shape_info = cls.polygon_functions[name](**params)
        vertices = np.array(shape_info.vertices, dtype=np.float32).tolist()

        result = dict(
            type='Polygon', vertices=vertices, rounding_radius=0)

        rmax = np.max(np.linalg.norm(vertices, axis=-1))
        result['circumsphere_radius'] = rmax
        result['rounding_volume_polynomial'] = shape_info.rounding_volume_polynomial

        return result

    @classmethod
    def get_shapedef(cls, name, params):
        """Get a previously-registered shape of any type by its name"""
        if name in cls.convex_polyhedron_functions:
            return cls.convex_polyhedron_shapedef(name, params)
        elif name in cls.polygon_functions:
            return cls.polygon_shapedef(name, params)
        else:
            raise NotImplementedError()

    @classmethod
    def verify_convex_polyhedra(cls):
        """Validate all registered convex polyhedra"""
        import scipy as sp, scipy.spatial

        for name in cls.convex_polyhedron_functions:
            shapedef = cls.get_shapedef(name, {})
            vertices = shapedef['vertices']
            hull = sp.spatial.ConvexHull(vertices)
            # detect coplanar simplices
            eqn_tree = sp.spatial.cKDTree(hull.equations)
            coplanar_facets = eqn_tree.query_pairs(1e-5)

            # convex polyhedra always have a full sphere's worth of
            # corner caps (4/3*pi*r**3). Last terms are just the
            # surface area and volume. Second (cylindrical wedges)
            # term will be computed from external edges below
            volume_polynomial = np.array([
                4./3*np.pi, 0, hull.area, hull.volume])

            # map point indices corresponding to an external edge to
            # the simplex indices that meet at that edge
            all_edges = {}
            # i, j are indices of facets
            for i, neighbors in enumerate(hull.neighbors):
                for j in neighbors:
                    ij = min(i, j), max(i, j)

                    if ij not in coplanar_facets:
                        # indices here are convex hull point indices
                        indices = set(hull.simplices[i])
                        indices.intersection_update(hull.simplices[j])

                        assert len(indices) == 2

                        edge_indices = min(indices), max(indices)
                        all_edges[edge_indices] = ij

            for (vi, vj), (si, sj) in all_edges.items():
                length = np.linalg.norm(hull.points[vi] - hull.points[vj])
                dot_product = np.dot(hull.equations[si, :3], hull.equations[sj, :3])
                angle = 2*np.pi - np.arccos(dot_product)

                volume_polynomial[1] += 0.5*length*angle

            volume = np.polyval(volume_polynomial, 0)

            try:
                assert np.isclose(hull.volume, volume)
                assert np.allclose(volume_polynomial, shapedef['rounding_volume_polynomial'])
            except AssertionError:
                print(shapedef)
                print('Computed volume, hull volume: {}, {}'.format(
                    volume, hull.volume))
                print('Computed volume polynomial: {}'.format(volume_polynomial))
                raise

    @classmethod
    def verify_polygons(cls):
        """Validate all registered polygons"""
        from hoomd.dem import utils

        for name in cls.polygon_functions:
            shapedef = cls.get_shapedef(name, {})
            vertices = shapedef['vertices']

            radii = [0.1, 1, 2, 3]
            volumes = [utils.spheroArea(vertices, r) for r in radii]

            volume_polynomial = np.polyfit(radii, volumes, 2)

            try:
                assert np.allclose(volume_polynomial, shapedef['rounding_volume_polynomial'])
            except AssertionError:
                print(shapedef)
                print('Computed volume polynomial: {}'.format(volume_polynomial))
                raise

@Shape.convex_polyhedron
def cube_shape():
    d = 0.5
    vertices = list(itertools.product([-d, d], [-d, d], [-d, d]))

    volume = 1
    surface_area = 6
    weighted_edge_length = 9*np.pi

    volume_poly = [4./3*np.pi, weighted_edge_length, surface_area, volume]

    return ShapeInfo(vertices, volume_poly)

@Shape.convex_polyhedron
def tetrahedron_shape():
    d = (8./3)**(-1./3)
    vertices = [(d, d, d), (d, -d, -d),
                (-d, d, -d), (-d, -d, d)]

    volume = 1
    surface_area = 6*3.**(1./6)
    weighted_edge_length = 26.75541247

    volume_poly = [4./3*np.pi, weighted_edge_length, surface_area, volume]

    return ShapeInfo(vertices, volume_poly)

@Shape.convex_polyhedron
def octahedron_shape():
    d = (4./3)**(-1./3)
    vertices = [(d, 0, 0), (-d, 0, 0),
                (0, d, 0), (0, -d, 0),
                (0, 0, d), (0, 0, -d)]

    volume = 1
    surface_area = 3*48**(1./6)
    weighted_edge_length = 38.94957767

    volume_poly = [4./3*np.pi, weighted_edge_length, surface_area, volume]

    return ShapeInfo(vertices, volume_poly)

@Shape.polygon
def regular_ngon_shape(n=3):
    vertex_distance = np.sqrt(2/n/np.sin(2*np.pi/n))
    thetas = np.linspace(0, 2*np.pi, n, endpoint=False)

    vertices = vertex_distance*np.array([np.cos(thetas), np.sin(thetas)]).T

    volume = 1
    surface_area = np.sqrt(4*n*np.tan(np.pi/n))

    volume_poly = [np.pi, surface_area, volume]

    return ShapeInfo(vertices, volume_poly)

def modify_shapedef(shape, modifications):
    for mod in modifications:
        mod_type = mod['type']
        if mod_type == 'scale':
            factor = mod['factor']
            shape['vertices'] = (factor*np.array(shape['vertices'])).tolist()
            shape['circumsphere_radius'] *= factor
            poly = shape['rounding_volume_polynomial']
            for (i, v) in enumerate(poly):
                poly[i] *= factor**i

        elif mod_type == 'round':
            radius = mod['radius']
            shape['circumsphere_radius'] += radius - shape.get('rounding_radius', 0)
            shape['rounding_radius'] = radius

        elif mod_type == 'unit_volume':
            dimensions = len(shape['vertices'][0])
            volume = np.polyval(
                shape['rounding_volume_polynomial'], shape.get('rounding_radius', 0))
            factor = volume**(-1./dimensions)
            shape = modify_shapedef(shape, [dict(type='scale', factor=factor)])

        else:
            raise NotImplementedError(mod_type)

    return shape

def make_shape(shape_params):
    shape_params = dict(shape_params)

    shape_name = shape_params.pop('type')
    modifications = shape_params.pop('modifications', [])

    base_shape = Shape.get_shapedef(shape_name, shape_params)
    true_shape = modify_shapedef(base_shape, modifications)

    return true_shape
