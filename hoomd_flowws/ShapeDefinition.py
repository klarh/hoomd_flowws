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
        scope['type_shapes'] = type_shapes

class Shape:
    """Helper class to make shape definitions more succinct"""
    convex_polyhedron_functions = {}

    @classmethod
    def convex_polyhedron(cls, function):
        name = function.__name__

        pattern = re.compile(r'^(?P<name>[a-zA-Z_]+)_shape$')
        match = pattern.match(name)

        assert match, 'convex_polyhedron-wrapped functions must be named <name>_shape'

        shape_type = match.group('name')

        cls.convex_polyhedron_functions[shape_type] = function
        return function

    @classmethod
    def convex_polyhedron_shapedef(cls, name, params):
        shape_info = cls.convex_polyhedron_functions[name](**params)
        vertices = np.array(shape_info.vertices, dtype=np.float32).tolist()

        result = dict(
            type='ConvexPolyhedron', vertices=vertices, rounding_radius=0)

        rmax = np.max(np.linalg.norm(vertices, axis=-1))
        result['circumsphere_radius'] = rmax
        result['rounding_volume_polynomial'] = shape_info.rounding_volume_polynomial

        return result

    @classmethod
    def get_shapedef(cls, name, params):
        if name in cls.convex_polyhedron_functions:
            return cls.convex_polyhedron_shapedef(name, params)
        else:
            raise NotImplementedError()

    @classmethod
    def verify_registered_shapes(cls):
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

def modify_shapedef(shape, modifications):
    for mod in modifications:
        mod_type = mod['type']
        if mod_type == 'scale':
            factor = mod['factor']
            shape['circumsphere_radius'] *= factor
            poly = shape['rounding_volume_polynomial']
            for (i, v) in enumerate(poly):
                poly[i] *= factor**i

        elif mod_type == 'round':
            radius = mod['radius']
            shape['circumsphere_radius'] += radius - shape.get('rounding_radius', 0)
            shape['rounding_radius'] = radius

        elif mod_type == 'unit_volume':
            volume = np.polyval(
                shape['rounding_volume_polynomial'], shape.get('rounding_radius', 0))
            factor = volume**(-1./3)
            shape = modify_shapedef(shape, [dict(type='scale', factor=factor)])

        else:
            raise NotImplementedError(mod_type)

    return shape

def make_shape(shape_params):
    shape_params = dict(shape_params)

    shape_name = shape_params.pop('name')
    modifications = shape_params.pop('modifications', [])

    base_shape = Shape.get_shapedef(shape_name, shape_params)
    true_shape = modify_shapedef(base_shape, modifications)

    return true_shape
