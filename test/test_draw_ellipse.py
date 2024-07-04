import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'deepvog')))
import unittest
import numpy as np
from deepvog.draw_ellipse import fit_ellipse

class TestDrawEllipse(unittest.TestCase):

    def test_fit_ellipse(self):
        test_data = np.load('test/test_data/testdata_draw_ellipse.npz', allow_pickle=True)
        
        test_inputs = test_data['input'][()]
        test_outputs = test_data['output'][()]
        rr, cc, center, w, h, radian, _ = fit_ellipse(**test_inputs)

        np.testing.assert_array_equal(rr, test_outputs['rr'])
        np.testing.assert_array_equal(cc, test_outputs['cc'])
        
        np.testing.assert_array_almost_equal(np.array(center), np.array(test_outputs['center']), decimal=6)
        self.assertAlmostEqual(w, test_outputs['w'])
        self.assertAlmostEqual(h, test_outputs['h'], )
        self.assertAlmostEqual(radian, test_outputs['radian'])


if __name__ == '__main__':
    unittest.main(verbosity=2)