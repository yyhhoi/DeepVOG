# test/test_math_operations.py
import unittest
import sys
import os
import numpy as np

# Add the lib directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'deepvog')))
from unprojection import unprojectGazePositions, convert_ell_to_general, reproject, reverse_reproject


class TestUnrojection(unittest.TestCase):

    def test_convert_ell_to_general(self):
        pred = convert_ell_to_general(2, 4, 6, 8, np.deg2rad(45))
        ans = (50, 28, 50, -312, -456, -1080)
        np.testing.assert_almost_equal(list(pred), list(ans), decimal=6)

    def test_unprojectGazePositions(self):
        ell_co=[3, 4, 2, 5, -7, 10]
        a, b, c, d = unprojectGazePositions([1, 2, 10], ell_co=ell_co, radius=20)
        a2 = np.array([[-0.52756007], [-0.66216149], [0.53218656]])
        b2 = np.array([[0.89576052], [0.40839032], [0.17558598]])
        c2 = np.array([[21.68390727], [-7.99477662], [33.33638646]])
        d2 = np.array([[ 15.77149648], [-12.44180151], [34.81768951]])
        np.testing.assert_array_almost_equal(a, a2, decimal=6)
        np.testing.assert_array_almost_equal(b, b2, decimal=6)
        np.testing.assert_array_almost_equal(c, c2, decimal=6)
        np.testing.assert_array_almost_equal(d, d2, decimal=6)

    def test_reproject(self):
        focal_length = 20
        x1 = np.array([[2], [4], [40]])
        y1 = np.array([[1], [2]])
        x2 = x1.squeeze()
        y2 = y1.squeeze()
        x3 = np.array([[1.5, 2.2, 10],[8.8, -5.6, 40]])
        y3 = np.array([[ 3.,   4.4],[ 4.4, -2.8]])
        pred1 = reproject(x1, focal_length=focal_length, batch_mode=False)
        pred2 = reproject(x2, focal_length=focal_length, batch_mode=False)
        pred3 = reproject(x3, focal_length=focal_length, batch_mode=True)
        np.testing.assert_array_almost_equal(pred1, y1, decimal=6)
        np.testing.assert_array_almost_equal(pred2, y2, decimal=6)
        np.testing.assert_array_almost_equal(pred3, y3, decimal=6)

    def test_reverse_reproject(self):
        focal_length = 12
        z = 16
        x1 = np.array([[2, 3]])
        x2 = np.array([[8], [-4]])
        x3 = np.array([[1.1, 24] ,[-5.5, 3.2]])
        y1 = [[2.66666667, 4.0]]
        y2 = [[10.66666667], [-5.33333333]]
        y3 = [[ 1.46666667, 32.0], [-7.33333333,  4.26666667]]
        pred1 = reverse_reproject(x1, z=z, focal_length=focal_length)
        pred2 = reverse_reproject(x2, z=z, focal_length=focal_length)
        pred3 = reverse_reproject(x3, z=z, focal_length=focal_length)
        np.testing.assert_array_almost_equal(pred1, y1, decimal=6)
        np.testing.assert_array_almost_equal(pred2, y2, decimal=6)
        np.testing.assert_array_almost_equal(pred3, y3, decimal=6)
        
if __name__ == '__main__':
    unittest.main()