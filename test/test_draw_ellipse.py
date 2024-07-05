import unittest
import numpy as np
from deepvog.draw_ellipse import fit_ellipse

class TestDrawEllipse(unittest.TestCase):

    def test_fit_ellipse(self):
        test_data = np.load('test/test_data/testdata_draw_ellipse.npz', allow_pickle=True)
        
        rr, cc, center, w, h, radian, _ = fit_ellipse(test_data['input_img'], 0.5, 'r', None)

        np.testing.assert_array_equal(rr, test_data['output_rr'])
        np.testing.assert_array_equal(cc, test_data['output_cc'])
        np.testing.assert_array_almost_equal(np.array(center), np.array(test_data['output_center']), decimal=6)
        self.assertAlmostEqual(w, test_data['output_w'])
        self.assertAlmostEqual(h, test_data['output_h'], )
        self.assertAlmostEqual(radian, test_data['output_radian'])


if __name__ == '__main__':
    unittest.main(verbosity=2)