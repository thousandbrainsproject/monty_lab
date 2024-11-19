# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors


def best_fit_transform(src, dst):
    """Calculates the least-squares best-fit transform.

    Maps corresponding points A to B in m spatial dimensions.

    Args:
        src: Nxm numpy array of corresponding points
        dst: Nxm numpy array of corresponding points

    Returns:
        transform, r_matrix, translation: rigid body transformation and associated
            params
    """
    assert src.shape == dst.shape

    # get number of dimensions
    m = src.shape[1]

    # translate points to their centroids
    centroid_a = np.mean(src, axis=0)
    centroid_b = np.mean(dst, axis=0)
    aa = src - centroid_a
    bb = dst - centroid_b

    # rotation matrix
    h_matrix = np.dot(aa.T, bb)
    u_matrix, _, vt = np.linalg.svd(h_matrix)
    r_matrix = np.dot(vt.T, u_matrix.T)

    # special reflection case
    if np.linalg.det(r_matrix) < 0:
        vt[m - 1, :] *= -1
        r_matrix = np.dot(vt.T, u_matrix.T)

    # translation
    translation = centroid_b.T - np.dot(r_matrix, centroid_a.T)

    # homogeneous transformation
    transform = np.identity(m + 1)
    transform[:m, :m] = r_matrix
    transform[:m, m] = translation

    return transform, r_matrix, translation


def nearest_neighbor_src_dst(src, dst):
    """Find the nearest (Euclidean) neighbor in dst for each point in src.

    Args:
        src: Nxm array of points
        dst: Nxm array of points

    Returns:
        distances: Euclidian distances of nearest neighbors of src in dst
        indices: Indices of nearest neighbors of src in dst
    """
    # assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)

    return distances.ravel(), indices.ravel()


def params_to_matrix_rigid(parameters):
    """Convert a 6 element vector into a rigid body transform matrix.

    Args:
        parameters: numpy array or similar with 6 entries

    Returns:
        Rigid body transform matrix
    """
    euler_angles = parameters[:3]
    translation = parameters[3:]
    matrix = np.zeros((4, 4))
    matrix[:3, :3] = Rotation.from_euler("xyz", euler_angles, degrees=True).as_matrix()
    matrix[:-1, -1] = translation

    return matrix


def params_to_matrix_rotation(parameters):
    return Rotation.from_euler("xyz", parameters[:3], degrees=True).as_matrix()


def matrix_to_params(matrix):
    """Convert a 4 x 4 matrix describing a rigid body transform into a 6d vector.

    Place rotation and translation parameters in the first 3 and last 3 entries
    respectively.

    Args:
        matrix: numpy array[4, 4]

    Returns:
        6d vector
    """
    parameters = np.zeros(6)
    euler_angles = Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz", degrees=True)
    translation = matrix[:-1, -1]

    parameters[:3] = euler_angles
    parameters[3:] = translation

    return parameters


def matrix_to_params_rotation(matrix):

    assert matrix.shape == (3, 3), "Input matrix must be 3 x 3"
    return Rotation.from_matrix(matrix).as_euler("xyz", degrees=True)
