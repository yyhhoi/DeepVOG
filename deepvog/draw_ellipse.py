"""
File: draw_ellise.py
Author: Yuk-Hoi Yiu
Description: 
    This script contain functions that find and fit an ellipse from an image, and return pixel indexes for drawing.
    The steps are outlined in Yiu et al., 2019 (Figure 3).
Usage:
    To fit an ellipse, use the function "fit_ellipse(img, threshold, color, mask)". 
"""

import numpy as np
import matplotlib as mpl
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from .bwperim import bwperim
from .ellipses import LSqEllipse #The code is pulled frm https://github.com/bdhammel/least-squares-ellipse-fitting
from skimage.draw import ellipse_perimeter
from numpy.typing import NDArray as npt

def isolate_islands(prediction: npt, threshold: float) -> npt:
    """The function takes in the pixel-wise classification results (a 2D image), closes small gaps, isolates connected regions,
    finds the region with the largest area, and determines such a region as the pupil ellipse.

    Parameters
    ----------
    prediction : ndarray
        (240, 320) with float [0, 1]. Pixel-wise classification probabilities of a pupile region.
    threshold : float
        Threshold between [0, 1] to decide if the pixel is part of the pupil.

    Returns
    -------
    output : ndarray
        A boolean image with shape (240, 320) with 1's marking the pupile region and 0's otherwise.
    """

    bw = closing(prediction > threshold , square(3))
    labelled = label(bw)  
    regions_properties = regionprops(labelled)
    max_region_area = 0
    select_region = 0
    for region in regions_properties:
        if region.area > max_region_area:
            max_region_area = region.area
            select_region = region
    output = np.zeros(labelled.shape)
    if select_region == 0:
        return output
    else:
        output[labelled == select_region.label] = 1
        return output

# input: output from bwperim -- 2D image with perimeter of the ellipse = 1
def gen_ellipse_contour_perim(perim: npt, color: str = "r") -> tuple | None: 
    """Compute ellipse parameters from a boolean image with 1's marking the ellipse perimenter and 0's otherwise.
    

    Parameters
    ----------
    perim : ndarray
        A boolean image of shape (240, 320) with 1's marking the ellipse perimenter and 0's otherwise.
        It should be the output from the function bwperim.
    color : str, optional
        Color to be drawn on the ellipse perimeter., by default "r"

    Returns
    -------
    rr, cc: ndarray
        indexes of pixels that form the ellipse perimeter, such that img[rr, cc] are the ellipse pixels
    center: list 
        [x0, y0] in the np indexing frame
    w, h: float
        major and minor axes of the ellipse
    radian: float
        Orientation of the ellipse (clockwise)
    ell: mpl.patches.Ellipse
        Object for plotting in matplotlib
    None: 
        If no ellipse is found, only None is output instead of the tuple (rr, cc, center, w, h, radian, ell).
    """
    # Vertices
    input_points = np.where(perim == 1)
    if (np.unique(input_points[0]).shape[0]) < 6 or (np.unique(input_points[1]).shape[0]< 6) :
        return None
    else:
        try:
            vertices = np.array([input_points[0], input_points[1]]).T
            # Contour
            fitted = LSqEllipse()
            fitted.fit([vertices[:,1], vertices[:,0]])
            center, w,h, radian = fitted.parameters()
            ell = mpl.patches.Ellipse(xy = [center[0],center[1]], width = w*2, height = h*2, angle = np.rad2deg(radian), fill = False, color = color)
            # Because of the np indexing of y-axis, orientation needs to be minus
            rr, cc = ellipse_perimeter(int(np.round(center[0])), int(np.round(center[1])), int(np.round(w)), int(np.round(h)), -radian)
            return (rr, cc, center, w, h, radian, ell)
        except:
            return None


def fit_ellipse(img: npt, threshold: float = 0.5, color: str = "r", mask: npt | None = None) -> tuple | None:
    """Fitting an ellipse to the pixels which form the largest connected area.

    Parameters
    ----------
    img : ndarray
        Pupil classification probabilities from the DeepVOG network. 
        Shape=(240, 320). float [0, 1]
    threshold : float, optional
        Thresold for the pixels to be classified as pupil, by default 0.5
    color : str, optional
        Pixel color for drawing the pupil ellipse boundary, by default "r"
    mask : ndarray or None, optional
        Mask to zero out low-confidence pixel/regions.
        If ndarray, a mask with float value [0, 1] with the same shape as img. Values < 0.5 will be assigned 0. 
        If None, the argument is ignored. By default None.

    Returns
    -------
    rr, cc: ndarray
        1d-array of indexes (int64) of pixels that form the ellipse perimeter, such that img[rr, cc] are the ellipse pixels
    center: list 
        [x0, y0] in the np indexing frame
    w, h: float
        major and minor axes of the ellipse
    radian: float
        Orientation of the ellipse (clockwise)
    ell: mpl.patches.Ellipse
        Object for plotting in matplotlib
    None: 
        If no ellipse is found, only None is output instead of the tuple (rr, cc, center, w, h, radian, ell).

    """    

    isolated_pred = isolate_islands(img, threshold = threshold)
    perim_pred = bwperim(isolated_pred)

    # masking eyelid away from bwperim_output. Currently not available in DeepVOG (But will be used in DeepVOG-3D)
    if mask is not None:
        perim_pred[mask < 0.5] = 0

    # masking bwperim_output on the img boundaries as 0 
    perim_pred[0, :] = 0
    perim_pred[perim_pred.shape[0]-1, :] = 0
    perim_pred[:, 0] = 0
    perim_pred[:, perim_pred.shape[1]-1] = 0
    ellipse_info = gen_ellipse_contour_perim(perim_pred, color)

    return ellipse_info


if __name__ == '__main__':

    test_prediction = np.load('deepvog/test_model_prediction.npy')

    rr, cc, center, w, h, radian, ell = fit_ellipse(test_prediction[0, :, :, 1])


    np.savez('test/test_data/testdata_draw_ellipse.npz', 
             input_img = test_prediction[0, :, :, 1],
             output_rr = rr,
             output_cc = cc,
             output_center = center,
             output_w = w,
             output_h = h,
             output_radian = radian,)