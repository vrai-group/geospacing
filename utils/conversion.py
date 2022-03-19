'''module with utils functions'''
import unittest
import numpy as np

def normal_to_dip_dip_direction(normals):
    """convert normal to get dip and dip direction"""
    n_sign = np.sign(normals[:, 2])
    dip_direction_rad = np.arctan2(n_sign * normals[:, 0], n_sign * normals[:, 1])
    dip_rad = np.arccos(np.abs(normals[:, 2]))
    sign_dip_rad = np.sign(dip_direction_rad)
    dip_direction_rad += np.pi * (1 - sign_dip_rad)
    dip = np.rad2deg(dip_rad)
    dip_direction = np.rad2deg(dip_direction_rad)
    return (dip, dip_direction)

def distance_point_plane(normal, point1, point2):
    "calculate distances between points and plane identified with point and normal"
    return np.dot(normal, point2 - point1)/np.linalg.norm(normal)

class Testing(unittest.TestCase):
    """Class for unit testing"""
    def test_distance_point_plane(self):
        """test distance point to plane"""
        point1 = np.array([0.0, 0.0, 0.0])
        point2 = np.array([0.0, 0.0, 1.0])
        normal = np.array([0.0, 0.0, 1.0])
        self.assertAlmostEqual(distance_point_plane(normal, point1, point2), 1.0)

    def test_normal_dip_dip_direction(self):
        """test normal to dip and dip direction"""
        normal = np.zeros((1, 3))
        normal[0, 2] = 1.0
        self.assertAlmostEqual(normal_to_dip_dip_direction(normal), (0.0, 180.0))

if __name__ == '__main__':
    unittest.main()
