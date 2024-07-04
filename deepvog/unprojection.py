# The unprojection algorithm is the work of Safaee-Rad et al. 1992. See https://ieeexplore.ieee.org/document/163786
# This python script is a re-implementation of Safaee-Rad et al.'s work.
# The algorithm unprojects a 2D ellipse in the image frame, to a circle in a 3D space.

import numpy as np
import logging
from numpy.typing import NDArray as npt

def _gen_cone_co(alpha: float, beta: float, gamma: float, a_prime: float, 
                h_prime: float, b_prime: float, g_prime: float, f_prime: float, d_prime: float) \
                -> tuple[float, float, float, float, float, float]:
    """
    This function converts the coefficients of the general ellipse equations into the general equation of a cone.
    Following Safaee-Rad et. al. 1992 equation (3):
    From:
        a'x^2 + 2h'xy + b'y^2 + 2g'x + 2f'y + d' = 0  and vertex (alpha, beta, gamma)
    To:
       ax^2 + by^2 + cz^2 + 2fyz + 2gzx + 2ux + 2vy + 2wz + d = 0 

    Parameters
    ----------
    alpha : float
        x-coordinate of the vertex point in the image frame.
    beta : float
        y-coordinate of the vertex point in the image frame.
    gamma : float
        z-coordinate of the vertex point in the image frame. Usually gamma = -e = focal length of the camera
    a_prime : float
        Coefficient of the ellipse equation (general form).
    h_prime, b_prime, g_prime, f_prime, d_prime : 
        Same as above.

    Returns
    -------
    tuple of floats
        Coefficients of the cone equation (general form).
    """
    gamma_square = np.power(gamma, 2)
    a = gamma_square * a_prime
    b = gamma_square * b_prime
    c = a_prime * np.power(alpha, 2) + 2 * h_prime * alpha * beta + b_prime * np.power(beta, 2) + 2 * g_prime * alpha + 2 * f_prime * beta + d_prime
    d = gamma_square * d_prime
    f = -gamma * (b_prime * beta + h_prime * alpha +f_prime)
    g = -gamma * (h_prime * beta + a_prime * alpha + g_prime)
    h = gamma_square * h_prime
    u = gamma_square * g_prime
    v = gamma_square * f_prime
    w = -gamma * (f_prime * beta + g_prime * alpha + d_prime)
    return a, b, c, d, f, g, h, u, v, w


def _gen_rotmat_co(lamb, a, b, g, f, h):
    """Safaee-Rad et. al., 1992 (8)
    """
    t1 = (b-lamb)*g - f*h
    t2 = (a - lamb)*f - g*h
    t3 = -(a-lamb) * (t1/t2)/g - (h/g)
    m = 1 / (np.sqrt(1 + np.power((t1/t2), 2) + np.power(t3, 2)))
    l = (t1/t2)*m
    n = t3*m
    return l, m, n

def _gen_lmn(lamb1, lamb2, lamb3):
    """Safaee-Rad et. al., 1992 (12), (27)-(33)
    """
    if lamb1 < lamb2:
        l = 0
        m_pos = np.sqrt((lamb2-lamb1)/(lamb2-lamb3))
        m_neg = -m_pos
        n = np.sqrt((lamb1-lamb3)/(lamb2-lamb3))
        return [l, l], [m_pos, m_neg], [n, n]
    elif lamb1 > lamb2:
        l_pos = np.sqrt((lamb1-lamb2)/(lamb1-lamb3))
        l_neg = -l_pos
        n = np.sqrt((lamb2-lamb3)/(lamb1-lamb3))
        m = 0
        return [l_pos, l_neg], [m, m], [n, n]
    elif lamb1 == lamb2:
        n = 1
        m = 0
        l = 0
        return [l,l], [m,m], [n,n]
    else:
        
        logging.warning("Failure to generate l,m,n. None's are returned")
        return None, None, None
    
def _calT3(l, m ,n):
    lm_sqrt = np.sqrt((l**2)+(m**2))
    T3 = np.array([-m/lm_sqrt, -(l*n)/lm_sqrt, l, 0,
                       l/lm_sqrt, -(m*n)/lm_sqrt, m, 0,
                       0, lm_sqrt, n, 0,
                       0, 0, 0, 1]).reshape(4,4)
    return T3

def _calABCD(T3, lamb1, lamb2, lamb3):
    li, mi, ni = T3[0:3,0], T3[0:3,1], T3[0:3,2]
    lamb_array = np.array([lamb1, lamb2, lamb3])
    A = np.dot(np.power(li,2), lamb_array)
    B = np.sum(li*ni*lamb_array)
    C = np.sum(mi*ni*lamb_array)
    D = np.dot(np.power(ni,2), lamb_array)
    return A,B,C,D

def _calXYZ_perfect(A, B, C, D, r):
    
    Z = (A*r)/np.sqrt((B**2)+(C**2)-A*D)
    X = (-B/A)*Z
    Y = (-C/A)*Z
    center = np.array([X,Y,Z,1]).reshape(4,1)
    return center
    
def check_parallel(v1, v2):
    a = np.dot(v1.T, v2)
    b = np.linalg.norm(v1) * np.linalg.norm(v2)
    radian = np.arccos(a/b).squeeze()
    return np.rad2deg(radian)

def convert_ell_to_general(xc: float, yc: float, w: float, h: float, radian: float) \
    -> tuple[float, float, float, float, float, float]:
    """Convert the parameters of an ellipse to the general equation's coefficients.

    The parameters xc, yc, w, h, radian corresond to the below equations of an ellipse:
    (X^2)/(w^2) + (X^2)/(h^2) = 1,
    where 
    X = (x-xc) * cos(radian) + (y-yc) * sin(radian)
    Y = -(x-xc) * sin(radian) + (y-yc) * cos(radian)

    To explain the above eqautions, the ellipse of major and minor axes (w, a) undergoes a translation of (xc, yc) and rotation by a "radian" degree (clockwise).

    Returns
    -------
    tuple[float * 6]
        Coefficients of the general ellipse equations.
        
    """
    A = (w**2) * (np.sin(radian)**2) + (h**2) * (np.cos(radian)**2)
    B = 2 * ((h**2) - (w**2)) * np.sin(radian) * np.cos(radian)
    C = (w**2) * (np.cos(radian)**2) + (h**2) * (np.sin(radian)**2)
    D = -2*A*xc - B*yc
    E = -B*xc - 2*C*yc
    F = A*(xc**2) + B*xc*yc + C*(yc**2) - (w**2) * (h**2)
    return A, B, C, D, E, F


def unprojectGazePositions(vertex: list | tuple, ell_co: list | tuple, radius: float) \
    -> tuple[npt, npt, npt, npt]:
    """The function takes in the vertex (x, y, z) coordiantes of the camera, parameters of the ellipse on the image plane, 
    and the assumed radius of the unprojected circular disk, and outputs the normal vectors of the unprojected circular disk in the 3D space (gaze vectors).

    The whole unrojection algorithm is documented in Safaee-Rad et al. 1992. Please refer to the article for details.

    Parameters
    ----------
    vertex : list or tuple
        x, y, z coordinates of the camera with respect to the image frame.
    ell_co : list or tuple
        6 coefficients of a generalised/expanded ellipse equations at the image frame.
        A*(x**2) + B*x*y + C*(y**2) + D*x + E*y + F = 0 (from https://en.wikipedia.org/wiki/Ellipse#General_ellipse)
    radius : float
        Assumed radius of the unprojected circular disk in the 3D space.

    Returns
    -------
    ndarray 
        Shape (3, 1). Positive normal vector of pupil disk in the camera frame.
    ndarray 
        Shape (3, 1). Negative normal vector of pupil disk in the camera frame.
    ndarray 
        Shape (3, 1). Pupil disk center (with positive norm) in the camera frame.
    ndarray 
        Shape (3, 1). Pupil disk center (with negative norm) in the camera frame.
    """

    
    # Coefficients of the general ellipse equation
    A, B, C, D, E, F = ell_co

    # Vertex (Point of the camera)
    alpha, beta, gamma = vertex
    
    # Ellipse parameter at image frame (z_c = +20) with respect to the camera frame
    # Safaee-Rad et. al. 1992 equation (1)
    a_prime = A
    h_prime = B / 2
    b_prime = C
    g_prime = D / 2
    f_prime = E / 2
    d_prime = F

    # Coefficients of the Cone at the image frame
    a, b, c, d, f, g, h, u, v, w = _gen_cone_co(alpha, beta, gamma, a_prime, h_prime, b_prime, g_prime, f_prime, d_prime)

    # Safaee-Rad, 1992 (10)
    lamb_co1 = 1
    lamb_co2 = - (a + b + c)
    lamb_co3 = (b*c + c*a + a*b - np.power(f,2) - np.power(g,2) - np.power(h,2))
    lamb_co4 = -(a*b*c + 2*f*g*h - a*np.power(f,2) - b*np.power(g,2) - c*np.power(h,2))
    lamb1, lamb2, lamb3 = np.roots([lamb_co1, lamb_co2, lamb_co3, lamb_co4])

    # generate Normal vector at the canonical frame
    l, m, n = _gen_lmn(lamb1,lamb2,lamb3)
    norm_cano_pos = np.array([l[0],m[0],n[0],1]).reshape(4,1)
    norm_cano_neg = np.array([l[1],m[1],n[1],1]).reshape(4,1)
    
    # T1 Rotational Transformation to the camera fream
    l1, m1, n1 = _gen_rotmat_co(lamb1, a,b,g,f,h)
    l2, m2, n2 = _gen_rotmat_co(lamb2, a,b,g,f,h)
    l3, m3, n3 = _gen_rotmat_co(lamb3, a,b,g,f,h)
    T1 = np.array([l1,l2,l3,0 ,m1, m2, m3,0, n1, n2, n3,0, 0,0,0,1]).reshape(4,4)
    li, mi, ni = T1[0,0:3], T1[1,0:3], T1[2,0:3]
    if np.cross(li,mi).dot(ni) < 0:
        li = -li
        mi = -mi
        ni = -ni
    T1[0,0:3], T1[1,0:3], T1[2,0:3] = li, mi, ni
    norm_cam_pos = np.dot(T1, norm_cano_pos)
    norm_cam_neg = np.dot(T1, norm_cano_neg)

    # Calculating T2
    T2 = np.eye(4)
    T2[0:3,3] = -(u*li+v*mi+w*ni)/np.array([lamb1, lamb2, lamb3])

    # Calculating T3
    T3_pos = _calT3(l[0], m[0], n[0])
    T3_neg = _calT3(l[1], m[1], n[1])

    # calculate ABCD
    A_pos, B_pos, C_pos, D_pos = _calABCD(T3_pos, lamb1, lamb2, lamb3)
    A_neg, B_neg, C_neg, D_neg = _calABCD(T3_neg, lamb1, lamb2, lamb3)

    # Calculating T0
    T0 = np.eye(4)
    T0[2,3] = -gamma # -gamma = -(vertex[2]) = -(-focal_length) = + focal_length

    # Calculating center position with respect to the perfect frame
    center_pos = _calXYZ_perfect(A_pos, B_pos, C_pos, D_pos, radius)
    center_neg = _calXYZ_perfect(A_neg, B_neg, C_neg, D_neg, radius)

    # From perfect frame to camera frame
    true_center_pos = np.matmul(T0,np.matmul(T1,np.matmul(T2,np.matmul(T3_pos,center_pos))))
    if true_center_pos[2] <0:
        center_pos[0:3] = -center_pos[0:3]
        true_center_pos = np.matmul(T0,np.matmul(T1,np.matmul(T2,np.matmul(T3_pos,center_pos))))
    true_center_neg = np.matmul(T0,np.matmul(T1,np.matmul(T2,np.matmul(T3_neg,center_neg))))
    if true_center_neg[2] <0:
        center_neg[0:3] = -center_neg[0:3]
        true_center_neg = np.matmul(T0,np.matmul(T1,np.matmul(T2,np.matmul(T3_neg,center_neg))))

    return norm_cam_pos[0:3], norm_cam_neg[0:3], true_center_pos[0:3], true_center_neg[0:3]

def reproject(vec_3d: npt, focal_length: npt, batch_mode:bool = False) -> npt:
    """Reproject 3-D coordinates back to the 2D plane in the camera frame. 
    The x, y coordinates are scaled by focal_length/z.

    Parameters
    ----------
    vec_3d : ndarray
        3-D coordinates to be projected. .Shape = (3, 1) or (3, ) if not batch_mode. (m, 3) if batch_mode.
    focal_length : float
        The focal length of the camera in mm.
    batch_mode : float
        If True, reproject a batch of 3-D vectors instead of just one.

    Returns
    -------
    ndarray
        x, y coordinates. Shape = (2, 1) or (2, ), or (m, 2) if batch_mode.
    """
    if batch_mode:
        vec_2d = (focal_length*(vec_3d[:,0:2]))/vec_3d[:,[2]]  # (m, 3) -> (m, 2)
    else:
        vec_2d = (focal_length*vec_3d[0:2])/vec_3d[2]  # (3, 1) -> (2, 1), or (3, ) -> (2, )
    return vec_2d
    
def reverse_reproject(vec_2d: npt, z: float, focal_length: float) -> npt:
    """Rescale the x and y coordinates when the 2-d vector is unprojected back to the 3-D space.


    Parameters
    ----------
    vec_2d : ndarray
        Input 2-D vector. Shape = (2, 1), (2, ) or (m, 2).
    z : float
        Z-coordiante of the unprojected 3-D vector.
    focal_length : float
        Camera focal length in mm.

    Returns
    -------
    ndarray
        Rescaled 2-D vector. Shape is the same as input.
    """
    # Scale the x,y in a reverse manner of reproject() function,
    # when you unproject the reprojected coordinate.
    vec_2d_scaled = (vec_2d*z)/focal_length
    return vec_2d_scaled


if __name__ == "__main__":
    
    pass