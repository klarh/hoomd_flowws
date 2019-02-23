import unittest

from hoomd_flowws.ShapeDefinition import Shape

class TestShapeProperties(unittest.TestCase):
    def test_shape_properties(self):
        Shape.verify_registered_shapes()

if __name__ == '__main__':
    unittest.main()
