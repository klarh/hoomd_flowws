import unittest

from hoomd_flowws.ShapeDefinition import Shape

class TestShapeProperties(unittest.TestCase):
    def test_convex_polyhedra(self):
        Shape.verify_convex_polyhedra()

    def test_polygons(self):
        Shape.verify_polygons()

if __name__ == '__main__':
    unittest.main()
