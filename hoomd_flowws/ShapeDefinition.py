import argparse
import collections
import itertools
import re

import numpy as np
import flowws
from flowws import Argument as Arg

import hoomd.dem

# Simple container for shape-generation functions to return
ShapeInfo = collections.namedtuple(
    'ShapeInfo', ['vertices', 'rounding_volume_polynomial'])

class ShapedefArgument(Arg):
    """Specialized Argument object to parse shape definition inputs"""

    SHAPE_ARG_DOC = """List of per-shape specifications and modifiers."""
    CMD_SHAPE_ARG_DOC = """Key-value pairs of shape arguments, specified for a given shape. To
    begin specifying a new shape, use `-a shape <shapeName>`. Valid
    arguments can include shape-specific arguments (if a shape take a
    num_vertices argument, for example, `-a num_vertices 6`) or generic
    arguments (`-a scale 2 -a round 0.25`)."""

    def __init__(self):
        super(ShapedefArgument, self).__init__(
            'shape_arguments', '-a', [{}], help=self.SHAPE_ARG_DOC,
            cmd_type=[(str, [eval])], cmd_help=self.CMD_SHAPE_ARG_DOC)

    def validate_cmd(self, value):
        # convert result into a list of dictionaries
        result = []

        for (key, vals) in value:
            if key == 'shape':
                result.append(dict(type=vals, modifications=[]))
            elif key == 'scale':
                result[-1]['modifications'].append(dict(type=key, factor=float(vals)))
            elif key == 'round':
                result[-1]['modifications'].append(dict(type=key, radius=float(vals)))
            elif key == 'unit_volume':
                result[-1]['modifications'].append(dict(type=key))
            else:
                result[-1][key] = eval(vals)

        return result

@flowws.add_stage_arguments
class ShapeDefinition(flowws.Stage):
    """Define per-type shapes for future stages to utilize

    Shape information is used for visualization, packing fraction
    calculations, and pair force/HPMC integrator configuration.

    Shapes consist of a base type, any parameters of the shape, and
    modifications. For example::

        # regular polygon with 4 vertices (square)
        shape = dict(type='regular_ngon', n=4,
                     modifications=[dict(type='scale', factor=2)])
        # rounded tetrahedron
        shape = dict(type='tetrahedron',
                     modifications=[dict(type='round', radius=0.5)])
        ShapeDefinition(shape_arguments=[shape])

    """
    ARGS = [
        ShapedefArgument(),
    ]

    def run(self, scope, storage):
        shape_arguments = self.arguments['shape_arguments']
        type_shapes = [make_shape(params) for params in shape_arguments]

        if type_shapes and len(type_shapes[0]['vertices'][0]) == 2:
            scope['dimensions'] = 2

        scope['type_shapes'] = type_shapes

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
                angle = np.arccos(dot_product)

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
def tetrahedron_shape():
    d = (8./3)**(-1./3)
    vertices = [(d, d, d), (d, -d, -d),
                (-d, d, -d), (-d, -d, d)]

    volume = 1
    surface_area = 6*3.**(1./6)
    weighted_edge_length = 11.69106268

    volume_poly = [4./3*np.pi, weighted_edge_length, surface_area, volume]

    return ShapeInfo(vertices, volume_poly)

@Shape.convex_polyhedron
def cube_shape():
    d = 0.5
    vertices = list(itertools.product([-d, d], [-d, d], [-d, d]))

    volume = 1
    surface_area = 6
    weighted_edge_length = 3*np.pi

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
    weighted_edge_length = 9.48994571

    volume_poly = [4./3*np.pi, weighted_edge_length, surface_area, volume]

    return ShapeInfo(vertices, volume_poly)

@Shape.convex_polyhedron
def dodecahedron_shape():
    phi = .5*(1 + np.sqrt(5))

    vertices = []
    for (a, b, i) in itertools.product([-1/phi, 1/phi], [-phi, phi], range(3)):
        vertices.append(np.roll((0, a, b), i))

    vertices.extend(list(itertools.product([-1, 1], [-1, 1], [-1, 1])))

    vertices = np.multiply(vertices, 14.472136405045545**(-1./3)).tolist()

    volume = 1
    surface_area = 5.3116139
    weighted_edge_length = 8.42355368

    volume_poly = [4./3*np.pi, weighted_edge_length, surface_area, volume]

    return ShapeInfo(vertices, volume_poly)

@Shape.convex_polyhedron
def icosahedron_shape():
    phi = .5*(1 + np.sqrt(5))

    vertices = []
    for (i, one_term, phi_term) in itertools.product(
            range(3), [-1, 1], [-phi, phi]):
        vertices.append(np.roll((0, one_term, phi_term), i))

    vertices = np.multiply(vertices, 17.453560309384446**(-1./3)).tolist()

    volume = 1
    surface_area = 5.14834876
    weighted_edge_length = 8.43957795

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
