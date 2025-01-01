# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


import numpy as np
import torch

from .pose_estimation_utils import params_to_matrix_rigid


class TransformedPointCloudDistance:
    """Functions are not serializable, make throwaway classes for now."""

    def __call__(self, **kwargs):
        """Compute pairwise distances of actual points and return the mean.

        Assume pc2 is exactly pc1 after applying some transformation,

        Args:
            **kwargs:
                dst: target point cloud batch := 1 x N_points x m_dimensions
                est: estimated target point cloud of the same shape

        Returns:
            Mean distance.
        """
        dst, est = kwargs["dst"], kwargs["est"]
        return self.distance(dst, est)

    @classmethod
    def distance(cls, a, b):
        return torch.norm(a - b, dim=2).mean()

    @classmethod
    def distance_no_batch(cls, a, b):
        return torch.norm(a - b, dim=1).mean()


class RigidBodyDisparity:
    """Return difference in euler angles.

    Assume params is estimating inverse of rotation.
    """

    def __call__(self, **kwargs):
        """Call the disparity function.

        Args:
            **kwargs:
                transform: vector of 3 euler angles + 3 translations of transform
                           applied to point cloud
                params: as above; represents the estimated inverse of rotation.

        Returns:
            rigid_body_disparity: 6d vector adding euler angles of rotation and
                translation params, representing any bias in how a point cloud is
                un-rotated
        """
        rot_1 = kwargs["transform"]
        rot_2 = kwargs["params"]

        return self.disparity(rot_1, rot_2)

    @classmethod
    def disparity(cls, rot_1, rot_2):
        rigid_body_dispairty = rot_1 + rot_2
        return rigid_body_dispairty


class InverseMatrixDeviation:

    def __call__(self, **kwargs):
        """Measure deviation of params * rotation from identity.

        Assume params is estimating inverse of rotation.

        Args:
            **kwargs:
                transform: vector of 3 euler angles of transform applied to point cloud,
                           possibly including 3 additional translation parameters
                params: as above, represents the estimated inverse of rotation

        Returns:
            deviation_from_identity: sum of (R^-1hat*R - I)^2

        Note:
            as_euler angles are in (-180, 180) for x and y, and (-90, 90) for z

        See Also:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_euler.html?highlight=euler#
        """
        transform = kwargs["transform"]
        params = kwargs["params"]

        return self.deviation(transform, params)

    @classmethod
    def deviation(cls, transform, params):

        assert len(transform) == 6, "expected 6 element vector"
        m1 = params_to_matrix_rigid(transform)
        m2 = params_to_matrix_rigid(params)

        identity_est = np.dot(m2, m1)
        deviation_from_identity = np.linalg.norm(np.eye(4) - identity_est)

        return deviation_from_identity


class PercentErrorReduction:
    """Map inputs to [0, 1], representing what percentage of possible error remains.

    0 means perfect, i.e. no error.

    Compute pointwise error reduction as
    error(est, dst) / (error(src, dst) - error(src_best, dst)), where
    error(src_best, dst) vanishes because src and dest are transformed versions pf one
    another. Thus simply compare error of estimate to real.
    """

    def __call__(self, **kwargs):

        src = kwargs["src"]
        dst = kwargs["dst"]
        est = kwargs["est"]

        return self.error_reduction(src, dst, est)

    @classmethod
    def error_reduction(cls, src, dst, est):

        raw_error = torch.norm(src - dst, axis=2).sum()
        est_error = torch.norm(est - dst, axis=2).sum()

        return est_error / raw_error
