import numpy as np
from numpy.typing import NDArray


class NoIntersectionError(Exception):
    pass

# vector (m,n) , m = number of examples, n = dimensionality
# a = coordinates of the vector
# n = orientation of the vector
def intersect(a: NDArray, n: NDArray) -> NDArray:
    """
    Calculate the intersection point of lines with a given normal vector.
    See eqution (2) in Yiu et al. 2019, and equation (6) in Swirski & Dodgson 2013

    Parameters
    ----------
    a : numpy.ndarray
        Array of shape (num_lines, dim) representing the pupil center xy position.
    n : numpy.ndarray
        Array of shape (num_lines, dim) representing the normal vectors.

    Returns
    -------
    numpy.ndarray
        Array of shape (dim, 1) representing the intersection point.

    Notes
    -----
    The function calculates the intersection point of lines defined by the line vectors `a`
    and the normal vectors `n`. The normal vectors are first normalized, and then the intersection
    point is calculated using the formula p = inv(R_sum) * q_sum, where R_sum is the sum of all
    rotation matrices and q_sum is the sum of all transformed line vectors.

    Examples
    --------
    >>> a = np.array([[1, 2, 3], [4, 5, 6]])
    >>> n = np.array([[0, 0, 1], [0, 1, 0]])
    >>> intersect(a, n)
    array([[2.5],
           [2.],
           [6.]])
    """
    # default normalisation of vectors n
    n = n/np.linalg.norm(n, axis=1, keepdims=True)
    num_lines = a.shape[0]
    dim = a.shape[1]
    I = np.eye(dim)
    R_sum = 0
    q_sum = 0
    for i in range(num_lines):    
        R = I - np.matmul(n[i].reshape(dim, 1), n[i].reshape(1, dim))
        q = np.matmul(R, a[i].reshape(dim, 1))
        q_sum = q_sum + q
        R_sum = R_sum + R
    p = np.matmul(np.linalg.inv(R_sum), q_sum)
    return p

def calc_distance(a: NDArray, n: NDArray, p: NDArray) -> float:
    """
    Calculates the average distance between a set of lines and a point. 
    The distance measure is the perpendicular euclidean distance between the line and the projection of the point on the line.

    Parameters:
    ----------
    a ndarray: 
        Shape = (Num_lines, dim). The center positions of the pupil ellipses (the origin of the line).
    n ndarray: 
        Shape = (Num_lines, dim). The normal vectors representing the orientation of the pupil ellipse.
    p ndarray: 
        Shape = (dim, ). A point in space, where the distance from its projection on the lines will be calculated and averaged.

    Returns:
    -------
    float: 
        The calculated average distance from p to the lines parameterized by (a, n)

    """
    num_lines = a.shape[0]
    dim = a.shape[1]
    I = np.eye(dim)
    D_sum = 0
    for i in range(num_lines):
        D_1 = (a[i].reshape(dim,1) - p.reshape(dim,1)).T
        D_2 = I - np.matmul(n[i].reshape(dim,1), n[i].reshape(1,dim))
        D_3 = D_1.T
        D = np.matmul(np.matmul(D_1,D_2),D_3)
        D_sum = D_sum + D
    D_sum = D_sum/num_lines
    return D_sum

def fit_ransac(a: NDArray, n: NDArray, max_iters: int =2000, samples_to_fit: int =20, min_distance : float=2000):
    """
    Fits a model using the RANSAC algorithm to find the best model. 
    In this case, the model is the intersection point of the lines (projected eyeball center on the 2D image plane).

    Parameters:
    ----------
    a : ndarray 
        Array of shape (num_lines, dim) representing the data points.
    n : ndarray 
        Array of shape (num_lines, dim) representing the normal vectors.
    max_iters : int, optional 
        Maximum number of iterations for the RANSAC algorithm. Defaults to 2000.
    samples_to_fit : int, optional
        Number of samples to fit the model. Defaults to 20.
    min_distance : float, optional 
        Minimum distance threshold for inliers. Defaults to 2000.

    Returns:
    -------
    ndarray: 
        Array of shape (dim, 1), Best model found by the RANSAC algorithm. 

    """
    num_lines = a.shape[0]
    
    best_model = None
    best_distance = min_distance
    for i in range(max_iters):
        sampling_index = np.random.choice(num_lines, size=samples_to_fit, replace=False)
        a_sampled = a[sampling_index, :]
        n_sampled = n[sampling_index, :]
        model_sampled = intersect(a_sampled, n_sampled)
        sampled_distance = calc_distance(a, n, model_sampled)
        
        if sampled_distance > min_distance:
            continue
        else:
            if sampled_distance < best_distance:
                best_model = model_sampled
                best_distance = sampled_distance
    
    return best_model

def line_sphere_intersect(c, r, o, u):
    """
    Calculates the intersection points between a line and a sphere.
    The equation is derived from the equation of a sphere and the parametric equation of a line, by solving:

    (1) || x - c**2 || = r**2
    (2) x = o + dl

    See https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection

    Parameters:
    ----------
    c : ndarray
        Centre of the eyeball (shape: (3,1))
    r : scalar
        Radius of the eyeball
    o : ndarray
        Origin of the line (shape: (3,1))
    u : ndarray
        Directional unit vector of the line (shape: (3,1))

    Returns:
    -------
    list
        A list containing the auxiliary variables of the parametrized line x = o + dl.
        The list contains two elements: [d1, d2]. The closer one (d2) to the camera should be chosen.
    """
    u = u/np.linalg.norm(u)
    delta = np.square(np.dot(u.T, (o-c))) - np.dot((o-c).T, (o-c)) + np.square(r)  # -> (1, 1)
    if delta < 0:
        raise NoIntersectionError
    else:
        tmp_term = -np.dot(u.T, (o-c))  # -> (1, 1)
        d1 = tmp_term + np.sqrt(delta)
        d2 = tmp_term - np.sqrt(delta)  # It is the closer one, since delta is positive
    # d1 can be removed. It was kept in Swirski & Dodgson 2013, though. So I keep it here in case problems arise.
    return [d1.squeeze(), d2.squeeze()]
    

#%%
if __name__ == "__main__":
    
    pass

    