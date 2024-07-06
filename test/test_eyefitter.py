import unittest
import numpy as np
from deepvog.eyefitter import SingleEyeFitter
from deepvog.model.DeepVOG_model import load_DeepVOG

class TestSingleEyeFitter(unittest.TestCase):

    def setUp(self):
        self.flen = 12
        self.ori_video_shape, self.sensor_size = np.array((240, 320)).squeeze(), np.array((3.6, 4.8)).squeeze()
        self.mm2px_scaling = np.linalg.norm(self.ori_video_shape) / np.linalg.norm(self.sensor_size)
        self.confidence_fitting_threshold = 0.96
        self.eyefitter = SingleEyeFitter(focal_length=self.flen * self.mm2px_scaling,
                                         pupil_radius=2 * self.mm2px_scaling,
                                         initial_eye_z=50 * self.mm2px_scaling,
                                         image_shape=(240, 320))

    def test_unproject_single_observation(self):

        
        test_data = np.load('test/test_data/eyefitter/testdata_unproject_single_observation.npz')
        gpos, gneg, cpos, cneg, (rr, cc, center, w, h, radian, confidence) = self.eyefitter.unproject_single_observation(test_data['input_prediction'])
        np.testing.assert_array_almost_equal(gpos, test_data['output_gpos'], decimal=6)
        np.testing.assert_array_almost_equal(gneg, test_data['output_gneg'], decimal=6)
        np.testing.assert_array_almost_equal(cpos, test_data['output_cpos'], decimal=6)
        np.testing.assert_array_almost_equal(cneg, test_data['output_cneg'], decimal=6)
        np.testing.assert_array_equal(rr, test_data['output_rr'])
        np.testing.assert_array_equal(cc, test_data['output_cc'])
        np.testing.assert_array_almost_equal(np.array(center), np.array(test_data['output_center']), decimal=6)
        self.assertAlmostEqual(w, test_data['output_w'])
        self.assertAlmostEqual(h, test_data['output_h'], )
        self.assertAlmostEqual(radian, test_data['output_radian'])
        self.assertAlmostEqual(confidence, test_data['output_confidence'])

    def test_estimate_eye_sphere(self):
        predictions = np.load('test/test_data/eyefitter/testdata_batched_subsampled_predictions.npy')
        
        vid_m = predictions.shape[0]
        for i in range(vid_m):
            _, _, _, _, ellipse_info = self.eyefitter.unproject_single_observation(predictions[i, ...])
            (rr, cc, centre, w, h, radian, ellipse_confidence) = ellipse_info
            if centre is not None:
                if (ellipse_confidence > self.confidence_fitting_threshold):
                    self.eyefitter.add_to_fitting()

        # RANSAC is not tested here. It is too noisy for comparing output values.
        _ = self.eyefitter.fit_projected_eye_centre(ransac=False)
        _, _ = self.eyefitter.estimate_eye_sphere()
        expected_projected_eye_centre = np.array([[129.73336792],[135.15884813]])
        expected_eye_centre = np.array([[-126.11096699], [  63.16186721], [3333.33333333]])
        expected_aver_eye_radius = 731.2955387299593
        np.testing.assert_array_almost_equal(self.eyefitter.projected_eye_centre, expected_projected_eye_centre, decimal=6)
        np.testing.assert_array_almost_equal(self.eyefitter.eye_centre, expected_eye_centre, decimal=6)
        self.assertAlmostEqual(self.eyefitter.aver_eye_radius, expected_aver_eye_radius)

        # # Test gaze estimation
        expected_gazeinfos = np.load('test/test_data/eyefitter/testdata_calc_gaze.npy')
        gazeinfos = np.zeros((vid_m, 8))
        for i in range(vid_m):
            _, _, _, _, ellipse_info = self.eyefitter.unproject_single_observation(predictions[i, ...])
            (rr, cc, centre, w, h, radian, ellipse_confidence) = ellipse_info
            if (centre is not None):
                pos_xyz, gaze_angles, gaze_vec, consistence = self.eyefitter.estimate_gaze()
                gazeinfos[i, :] = np.array(list(pos_xyz) + centre + list(gaze_angles) + [consistence * 1.0])
        np.testing.assert_array_almost_equal(gazeinfos, expected_gazeinfos, decimal=6)

if __name__ == '__main__':
    unittest.main(verbosity=2)