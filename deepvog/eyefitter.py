import numpy as np
from .draw_ellipse import fit_ellipse
from .unprojection import convert_ell_to_general, unprojectGazePositions, reproject, reverse_reproject
from .intersection import NoIntersectionError, intersect, fit_ransac, line_sphere_intersect
from .CheckEllipse import computeEllipseConfidence
from .utils import convert_vec2angle31
"""
Unless specified, all units are in pixels. 
All calculations are in camera frame (conversion would be noted in comment)

"""


class SingleEyeFitter(object):

    def __init__(self, focal_length, pupil_radius, initial_eye_z, image_shape=(240, 320)):
        self.focal_length = focal_length
        self.image_shape = image_shape

        self.pupil_radius = pupil_radius
        self.vertex = [0, 0, -focal_length]
        self.initial_eye_z = initial_eye_z

        # (p,n) of unprojected gaze vector and pupil 3D position in SINGLE OBSERVATION
        self.current_gaze_pos = 0  # reserved for (3,1) np.array in camera frame
        self.current_gaze_neg = 0  # reserved for (3,1) np.array in camera frame
        self.current_pupil_3Dcentre_pos = 0  # reserved for (3,1) np.array in camera frame
        self.current_pupil_3Dcentre_neg = 0  # reserved for (3,1) np.array in camera frame
        self.current_ellipse_centre = 0  # reserved for numpy array (2,1) in numpy indexing frame

        # List of parameters across a number (m) of observations
        self.unprojected_gaze_vectors = []  # A list: ["gaze_positive"~np(m,3), "gaze_negative"~np(m,3)]
        self.unprojected_3D_pupil_positions = []  # [ "pupil_3Dcentre_positive"~np(m,3), "pupil_3Dcentre_negative"~np(m,3) ]
        self.ellipse_centres = None  # reserved for numpy array (m,2) in numpy indexing frame,
        # m = number of fitted ellipse centres corresponding to the projected gaze lines
        self.selected_gazes = None  # reserved for (m,3) np.array in camera frame
        self.selected_pupil_positions = None  # reserved for (m,3) np.array in camera frame

        # Parameters of the eye model for consistent pupil estimate after initialisation
        self.projected_eye_centre = None  # reserved for numpy array (2,1). Centre coordinate in numpy indexing frame.
        self.eye_centre = None  # reserved for (3,1) numpy array. 3D centre coordinate in camera frame
        self.aver_eye_radius = None  # Scaler


    def unproject_single_observation(self, prediction, mask=None, threshold=0.5):
        # "prediction" is an numpy array with shape (image_height, image_width) and data type: float [0-1]
        # Our deeplearning model's outpupt is Y~(240, 320, 3),
        # you will have to slice it manually as Y[:,:,1] as input to this function.
        try:
            assert len(prediction.shape) == 2
            assert prediction.shape == self.image_shape
        except(AssertionError):
            raise AssertionError(
                "Shape of the observation input has to be (image_height, image_width) specified in the initialization of object, or if default, (240,320)")
        # Fit an ellipse from the prediction map
        ellipse_info = fit_ellipse(prediction, mask=mask)
        rr, cc, centre, w, h, radian = None, None, None, None, None, None
        ellipse_confidence = 0

        # We unproject the gaze vectors and pupil centre only if an ellipse has been detected
        if ellipse_info is not None:

            (rr, cc, centre, w, h, radian, ell) = ellipse_info
            ellipse_confidence = computeEllipseConfidence(prediction, centre, w, h, radian)

            # Convert centre coordinates from numpy indexing frame to camera frames
            centre_cam = centre.copy()
            centre_cam[0] = centre_cam[0] - self.image_shape[1] / 2
            centre_cam[1] = centre_cam[1] - self.image_shape[0] / 2

            # Convert ellipse parameters to the coefficients of the general form of ellipse equation
            A, B, C, D, E, F = convert_ell_to_general(centre_cam[0], centre_cam[1], w, h, radian)
            ell_co = (A, B, C, D, E, F)

            # Unproject the ellipse to obtain 2 ambiguous gaze vectors with numpy shape (3,1),
            # and pupil_centre with numpy shape (3,1)
            unprojected_gaze_pos, unprojected_gaze_neg, unprojected_pupil_3Dcentre_pos, unprojected_pupil_3Dcentre_neg = unprojectGazePositions(
                self.vertex, ell_co, self.pupil_radius)

            # Normalize the gaze vectors and only take their real component
            unprojected_gaze_pos = unprojected_gaze_pos / np.linalg.norm(unprojected_gaze_pos)
            unprojected_gaze_neg = unprojected_gaze_neg / np.linalg.norm(unprojected_gaze_neg)

            unprojected_gaze_pos, unprojected_gaze_neg, unprojected_pupil_3Dcentre_pos, unprojected_pupil_3Dcentre_neg = np.real(
                unprojected_gaze_pos), np.real(unprojected_gaze_neg), np.real(unprojected_pupil_3Dcentre_pos), np.real(
                unprojected_pupil_3Dcentre_neg)
            self.current_gaze_pos, self.current_gaze_neg, self.current_pupil_3Dcentre_pos, self.current_pupil_3Dcentre_neg = unprojected_gaze_pos, unprojected_gaze_neg, unprojected_pupil_3Dcentre_pos, unprojected_pupil_3Dcentre_neg
            self.current_ellipse_centre = np.array(centre).reshape(2, 1)
        else:
            self.current_gaze_pos, self.current_gaze_neg, self.current_pupil_3Dcentre_pos, self.current_pupil_3Dcentre_neg = None, None, None, None
            self.current_ellipse_centre = None

        return self.current_gaze_pos, self.current_gaze_neg, self.current_pupil_3Dcentre_pos, self.current_pupil_3Dcentre_neg, (
        rr, cc, centre, w, h, radian, ellipse_confidence)

    def add_to_fitting(self):
        # Append parameterised gaze lines for fitting
        if (self.current_gaze_pos is None) or (self.current_gaze_neg is None) or (
                self.current_pupil_3Dcentre_pos is None) or (self.current_pupil_3Dcentre_neg is None) or (
                self.current_ellipse_centre is None):
            raise TypeError(
                'No ellipse was caught in this observation, thus "None" is being added for fitting set, which is not allowed. Please manually skip this condition.')

        # Store the gaze vectors and pupil 3D centres
        if (len(self.unprojected_gaze_vectors) == 0) or (len(self.unprojected_3D_pupil_positions) == 0) or (
                self.ellipse_centres is None):
            self.unprojected_gaze_vectors.append(self.current_gaze_pos.reshape(1, 3))
            self.unprojected_gaze_vectors.append(self.current_gaze_neg.reshape(1, 3))
            self.unprojected_3D_pupil_positions.append(self.current_pupil_3Dcentre_pos.reshape(1, 3))
            self.unprojected_3D_pupil_positions.append(self.current_pupil_3Dcentre_neg.reshape(1, 3))
            self.ellipse_centres = self.current_ellipse_centre.reshape(1, 2)
        else:
            self.unprojected_gaze_vectors[0] = np.vstack(
                (self.unprojected_gaze_vectors[0], self.current_gaze_pos.reshape(1, 3)))
            self.unprojected_gaze_vectors[1] = np.vstack(
                (self.unprojected_gaze_vectors[1], self.current_gaze_neg.reshape(1, 3)))
            self.unprojected_3D_pupil_positions[0] = np.vstack(
                (self.unprojected_3D_pupil_positions[0], self.current_pupil_3Dcentre_pos.reshape(1, 3)))
            self.unprojected_3D_pupil_positions[1] = np.vstack(
                (self.unprojected_3D_pupil_positions[1], self.current_pupil_3Dcentre_neg.reshape(1, 3)))
            self.ellipse_centres = np.vstack((self.ellipse_centres, self.current_ellipse_centre.reshape(1, 2)))

    def fit_projected_eye_centre(self, ransac=False, max_iters=1000, min_distance=2000):
        # You will need to determine when to fit outside of the class
        if (self.unprojected_gaze_vectors is None) or (self.ellipse_centres is None):
            msg = "No unprojected gaze lines or ellipse centres were found (not yet initalized). It is likely that the network fails to segment the pupil from the video. Please ensure your input video contains only a single eye but not other facial/body features."
            raise TypeError(msg)

        # Combining positive and negative gaze vectors
        a = np.vstack((self.ellipse_centres, self.ellipse_centres))
        n = np.vstack((self.unprojected_gaze_vectors[0][:, 0:2],
                       self.unprojected_gaze_vectors[1][:, 0:2]))  # [:, 0:2] takes only 2D projection

        # Normalisation of the 2D projection of gaze vectors is done inside intersect()
        if ransac == True:
            samples_to_fit = np.ceil(a.shape[0]/5).astype(np.int)  # Assuming 20% of outliners
            self.projected_eye_centre = fit_ransac(a, n, max_iters=max_iters, samples_to_fit=samples_to_fit,
                                                   min_distance=min_distance)
        else:
            self.projected_eye_centre = intersect(a, n)
        if (self.projected_eye_centre is None):
            raise TypeError("Projected_eye_centre was not fitted. You may need -v and -m argument to check whether the pupil segmentation works properly.")
        return self.projected_eye_centre

    def estimate_eye_sphere(self):
        # This function is called once after fit_projected_eye_centre()
        # self.initial_eye_z is required (in pixel unit)
        # self.initial_eye_z shall be the z-distance between the point and camera vertex (in camera frame)
        if (self.projected_eye_centre is None):
            # pdb.set_trace()
            raise TypeError('Projected_eye_centre must be initialized first')

        # Unprojecting the 2D projected eye centre to 3D.
        # Converting the projected_eye_centre from numpy indexing frame to camera frame
        projected_eye_centre_camera_frame = self.projected_eye_centre.copy()
        projected_eye_centre_camera_frame[0] = projected_eye_centre_camera_frame[0] - self.image_shape[1] / 2
        projected_eye_centre_camera_frame[1] = projected_eye_centre_camera_frame[1] - self.image_shape[0] / 2

        # Unprojection: Nearest intersection of two lines. 
        # a = [eye_centre, pupil_3Dcentre], n =[gaze_vector, pupil_3D_centre]
        projected_eye_centre_camera_frame_scaled = reverse_reproject(projected_eye_centre_camera_frame,
                                                                     self.initial_eye_z, self.focal_length)
        eye_centre_camera_frame = np.append(projected_eye_centre_camera_frame_scaled, self.initial_eye_z).reshape(3, 1)

        # Reconstructed selected gaze vectors and pupil positions by rejecting those pointing away from projected eyecentre
        m = self.unprojected_gaze_vectors[0].shape[0]
        for i in range(m):
            gazes = [self.unprojected_gaze_vectors[0][i, :].reshape(3, 1),
                     self.unprojected_gaze_vectors[1][i, :].reshape(3, 1)]
            positions = [self.unprojected_3D_pupil_positions[0][i, :].reshape(3, 1),
                         self.unprojected_3D_pupil_positions[1][i, :].reshape(3, 1)]
            selected_gaze, selected_position = self.select_pupil_from_single_observation(gazes, positions,
                                                                                         eye_centre_camera_frame)

            self.selected_gazes, self.selected_pupil_positions = self.stacking_from_nx1_to_mxn(
                [self.selected_gazes, self.selected_pupil_positions],
                [selected_gaze, selected_position],
                [3, 3])

        radius_counter = []
        for i in range(self.selected_gazes.shape[0]):
            gaze = self.selected_gazes[i, :].reshape(1, 3)
            position = self.selected_pupil_positions[i, :].reshape(1, 3)

            # Before stacking, you must reshape (3,1) to (1,3)
            a_3Dfitting = np.vstack((eye_centre_camera_frame.reshape(1, 3), position))
            n_3Dfitting = np.vstack((gaze, (position / np.linalg.norm(position))))

            intersected_pupil_3D_centre = intersect(a_3Dfitting, n_3Dfitting)
            radius = np.linalg.norm(intersected_pupil_3D_centre - eye_centre_camera_frame)
            radius_counter.append(radius)
        aver_radius = np.mean(radius_counter)

        self.aver_eye_radius = aver_radius
        self.eye_centre = eye_centre_camera_frame
        return aver_radius, radius_counter

    def estimate_gaze(self):
        # This function must be called after using unproject_single_observation() to update current observation
        if (self.eye_centre is None) or (self.aver_eye_radius is None):
            raise TypeError("Call estimate_eye_sphere() to initialize eye_centre and eye_radius first.")

        selected_gaze, selected_position = self.select_pupil_from_single_observation(
            [self.current_gaze_pos, self.current_gaze_neg],
            [self.current_pupil_3Dcentre_pos, self.current_pupil_3Dcentre_neg], self.eye_centre)
        

        # Perform line-sphere intersection to find the pupil position, radius, gaze vector
        try:
            o = np.zeros((3, 1))
            d1, d2 = line_sphere_intersect(self.eye_centre, self.aver_eye_radius, o,
                                            selected_position / np.linalg.norm(selected_position))
            d = min([d1, d2])  # Choose the minimum intersection (closest to the camera)

            pos = o + d * (selected_position / np.linalg.norm(selected_position))
            radius = (self.pupil_radius / selected_position[2, 0]) * pos[2, 0]
            gaze_vec = pos - self.eye_centre
            gaze_vec = gaze_vec / np.linalg.norm(gaze_vec)
            consistence = True

        except(NoIntersectionError):
            # print("Cannot find line-sphere interception. Old pupil parameters are used.")
            pos = selected_position
            gaze_vec = selected_gaze
            radius = self.pupil_radius
            consistence = False

        pos_xyz = (pos[0, 0], pos[1, 0], pos[2, 0])  # -> tuple, (x, y, z)
        gaze_angles = convert_vec2angle31(gaze_vec)  # -> tuple, (x_angle, y_angle)
        return pos_xyz, gaze_angles, gaze_vec, consistence



    def plot_gaze_lines(self, ax):
        t = np.linspace(-1000, 1000, 1000)
        a = np.vstack((self.ellipse_centres, self.ellipse_centres))
        n = np.vstack((self.unprojected_gaze_vectors[0][:, 0:2],
                       self.unprojected_gaze_vectors[1][:, 0:2]))  # [:, 0:2] takes only 2D projection

        for i in range(a.shape[0]):
            a_each = a[i, :]
            n_each = n[i, :]

            points = np.array(a_each).reshape(2, 1) + (t * n_each[0:2].reshape(2, 1))
            ax.plot(points[0, :], points[1, :])
        ax.set_xlim(0, self.image_shape[1])
        ax.set_ylim(self.image_shape[0], 0)
        return ax

    def select_pupil_from_single_observation(self, gazes, positions, eye_centre_camera_frame):
        # gazes is a list ~ [gaze_vector_pos~(3,1), gaze_vector_neg~(3,1)]
        # positions is a list ~ [pupil_position_pos~(3,1), pupil_position_neg~(3,1)]
        # eye_centre_camera_frame ~ numpy array~(3,1)

        selected_gaze = gazes[0]
        selected_position = positions[0]
        projected_centre = reproject(eye_centre_camera_frame, self.focal_length)
        projected_gaze = reproject(selected_position + selected_gaze, self.focal_length) - projected_centre
        projected_position = reproject(selected_position, self.focal_length)
        if np.dot(projected_gaze.T, (projected_position - projected_centre)) > 0:
            return selected_gaze, selected_position
        else:
            return gazes[1], positions[1]

    @staticmethod
    def stacking_from_nx1_to_mxn(stacked_arrays_list, stacked_vectors_list, dims_list):
        list_as_array = np.array([stacked_arrays_list])
        new_stacked_arrays_list = []
        if np.all(list_as_array == None):
            for stacked_array, stacked_vector, n in zip(stacked_arrays_list, stacked_vectors_list, dims_list):
                stacked_array = stacked_vector.reshape(1, n)
                new_stacked_arrays_list.append(stacked_array)
        elif np.all(list_as_array != None):
            for stacked_array, stacked_vector, n in zip(stacked_arrays_list, stacked_vectors_list, dims_list):
                stacked_array = np.vstack((stacked_array, stacked_vector.reshape(1, n)))
                new_stacked_arrays_list.append(stacked_array)
        elif np.any(list_as_array == None):
            print("Error list =\n", stacked_arrays_list)
            raise TypeError("Some lists are initialized, some are not ('None'). Error has happened!")
        else:
            print("Error list =\n", stacked_arrays_list)
            raise TypeError("Unknown Error Occurred.")
        return new_stacked_arrays_list




if __name__ == '__main__':

    flen = 12
    ori_video_shape, sensor_size = np.array((240, 320)).squeeze(), np.array((3.6, 4.8)).squeeze()
    mm2px_scaling = np.linalg.norm(ori_video_shape) / np.linalg.norm(sensor_size)
    confidence_fitting_threshold = 0.96
    eyefitter = SingleEyeFitter(focal_length=flen * mm2px_scaling,
                                        pupil_radius=2 * mm2px_scaling,
                                        initial_eye_z=50 * mm2px_scaling,
                                        image_shape=(240, 320))
    
    # # Test data for test_unproject_single_observation
    # test_data = np.load('test/test_data/test_model_prediction.npy')
    # prediction = test_data[0, :, :, 1]
    # gpos, gneg, cpos, cneg, (rr, cc, center, w, h, radian, confidence) = eyefitter.unproject_single_observation(prediction)
    # np.savez('test/test_data/testdata_unproject_single_observation.npz', 
    #          input_prediction = prediction,
    #          output_gpos=gpos,
    #          output_gneg=gneg,
    #          output_cpos=cpos,
    #          output_cneg=cneg,
    #          output_rr=rr, output_cc=cc, output_center=center, output_w=w, output_h=h, output_radian=radian, output_confidence=confidence)


    # # Test data for test_estimate_eye_sphere
    predictions = np.load('test/test_data/testdata_batched_subsampled_predictions.npy')
    
    vid_m = predictions.shape[0]
    for i in range(vid_m):
        _, _, _, _, ellipse_info = eyefitter.unproject_single_observation(predictions[i, ...])
        (rr, cc, centre, w, h, radian, ellipse_confidence) = ellipse_info
        if centre is not None:
            if (ellipse_confidence > confidence_fitting_threshold):
                eyefitter.add_to_fitting()
    # _ = eyefitter.fit_projected_eye_centre(ransac=True, max_iters=5000, min_distance=10*vid_m*20)
    _ = eyefitter.fit_projected_eye_centre(ransac=False)
    _, _ = eyefitter.estimate_eye_sphere()
    print('Projected eye center\n', eyefitter.projected_eye_centre)
    print('3D Eye center\n', eyefitter.eye_centre)
    print('Eye radius\n', eyefitter.aver_eye_radius)

    # # Test data for test_calc_gaze
    testdata = np.zeros((vid_m, 8))
    for i in range(vid_m):
        _, _, _, _, ellipse_info = eyefitter.unproject_single_observation(predictions[i, ...])
        (rr, cc, centre, w, h, radian, ellipse_confidence) = ellipse_info
        if (centre is not None):
            p_list, n_list, _, consistence = eyefitter.gen_consistent_pupil()
            positions, gaze_angles = eyefitter.calc_gaze(p_list=p_list, n_list=n_list)
            testdata[i, :] = np.array(list(positions) + centre + list(gaze_angles) + [consistence * 1.0])
    np.save('test/test_data/testdata_calc_gaze.npy', testdata)





    