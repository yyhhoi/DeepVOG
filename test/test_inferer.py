import unittest
import numpy as np
from deepvog.utils import load_json, csv_reader
import deepvog
import uuid
import os

class TestGazeInferer(unittest.TestCase):

    def setUp(self):
        model = deepvog.load_DeepVOG()

        focal_length = 12
        video_shape = (240, 320)
        sensor_size = (3.6, 4.8)
        
        self.inferer = deepvog.GazeInferer(model, focal_length, video_shape, sensor_size) 
        self.input_video = 'test/test_data/inferer/testdata_subsampled_video.mp4'
        self.expected_eyeball_model = 'test/test_data/inferer/testdata_fitted_eyeball_model.json'
        self.expected_gaze_results = csv_reader('test/test_data/inferer/testdata_gaze_results.csv')

    def test_inferer_process_fit(self):
        # Test that 
        #   (1) the fitted eyeball result can be successfully saved. i.e. inferer.save_eyeball_model() 
        #   (2) the fitted eyeball result is the same as the expected result. i.e. inferer.process(..., mode='Fit', ...)
        self.inferer.process(self.input_video, mode="Fit", batch_size=2, ransac=False)
        tmp_eyeball_pth = '%s.json'% str(uuid.uuid4())
        self.inferer.save_eyeball_model(tmp_eyeball_pth)
        expected_eyeball_model = load_json(self.expected_eyeball_model)
        predicted_eyeball_model = load_json(tmp_eyeball_pth)
        np.testing.assert_array_almost_equal(np.array(predicted_eyeball_model['eye_centre']), np.array(expected_eyeball_model['eye_centre']), decimal=6)
        self.assertAlmostEqual(predicted_eyeball_model['aver_eye_radius'], expected_eyeball_model['aver_eye_radius'])
        if os.path.exists(tmp_eyeball_pth):
            os.remove(tmp_eyeball_pth)

    def test_inferer_process_infer(self):
        # Test that 
        #   (1) the fitted eyeball result can be successfully loaded. i.e. inferer.load_eyeball_model() 
        #   (2) The gaze inference result is the same as expected. i.e. inferer.process(..., mode='Infer', ...)
        #   (3) A visualization video can be produced, but its content will not be tested. 
        #       The produced video will persist in the file system for visual inspection.
        self.inferer.load_eyeball_model(self.expected_eyeball_model)

        tmp_gaze_result_pth = '%s.csv'%(uuid.uuid4())
        self.inferer.process(self.input_video, mode="Infer",
                             output_record_path=tmp_gaze_result_pth,
                            output_video_path="test_visualization.mp4", batch_size=2)
        self.inferer.result_recorder.text_writer.close()
        predicted_gaze_results = csv_reader(tmp_gaze_result_pth)

        self.assertCountEqual(predicted_gaze_results.keys(), self.expected_gaze_results.keys())
        
        for key in self.expected_gaze_results.keys():
            np.testing.assert_array_almost_equal(np.array(predicted_gaze_results[key]).astype(float), 
                                                 np.array(self.expected_gaze_results[key]).astype(float), decimal=6)

        if os.path.exists(tmp_gaze_result_pth):
            os.remove(tmp_gaze_result_pth)

if __name__ == '__main__':
    unittest.main(verbosity=2)