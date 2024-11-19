# Copyright 2021-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import numpy as np
import torch
from scipy.spatial.transform import Rotation


class RandomOcclusion:
    """Transformation class that applies a random occlusion to a 3D pointcloud.

    The random occlusion occurs by selecting a random 2D plane to slice the 3D
    pointcloud such that a certain fraction of the pointcloud is occluded. Since
    the shape of the `__call__` method's output tensor is the same as its input, all
    (x, y, z) coordinates on the occluded side of the 2D plane are set equal to some
    point on the unoccluded side.

    Attributes:
        percent_to_occlude: the fraction of a 3D pointcloud to occlude
    """

    def __init__(self, percent_to_occlude):
        self.percent_to_occlude = percent_to_occlude

    def __call__(self, x):
        """Returns an occluded version of `x`.

        The occluded points/rows in this tensor are replaced with unoccluded
        points/rows. The random occlusion is determined on-the-fly and likely to be
        different for each call to this method.

        Args:
            x: PyTorch Tensor with shape (number of points, 3) representing a 3D
               pointcloud

        Returns:
            x, same as input, with a random occlusion applied
        """
        num_points, _ = x.size()

        # Generate a random 2D plane to slice each batch item; each such 2D plane is
        # defined the equation ax + by + cz + d = 0 and these a, b, c, and d values will
        # be stored in `coeffs`

        # First only generate the a, b, and c values since they define the orientation
        # of the plane
        coeffs = torch.rand(3)
        coeffs = coeffs * 2.0 - 1.0
        coeffs = coeffs.to(x.device)
        inner_prods = (x * coeffs.unsqueeze(0)).sum(dim=1)

        # `inner_prods` stores the values of ax + by + cz of each point in the
        # pointcloud, and each such value tells how far a point is from the pointcloud;
        # the furthest points will certainly be part of the occluded pointcloud, so if
        # we rank-order `inner_prods`, we can pick the points corresponding to the
        # desired occlusion level
        k = int((1.0 - self.percent_to_occlude) * num_points)
        values, inds = torch.topk(inner_prods, k=k)

        # Pick the appropriate value of d (which shifts the 2D plane towards or away
        # from its normal vector) to determine where to slice the pointcloud given a
        # fixed orientation of the 2D plane as defined by `coeffs`
        threshold = values.min().unsqueeze(0)
        d = -1.0 * threshold
        coeffs = torch.cat((coeffs, d))

        # Create a mask with shape (num_points,) where entry i is True iff the ith point
        # in `x` is part of in the occluded pointcloud, False otherwise
        mask = torch.zeros(num_points, dtype=torch.bool)
        mask[inds] = True
        included_points = x[mask, :]

        # For each occluded point that needs to be replaced, pick a random point on the
        # unoccluded side to replace it
        for n in range(mask.size(0)):
            if not mask[n]:
                i = np.random.randint(low=0, high=included_points.size(0))
                x[n, :] = included_points[i, :]

        return x


class RandomRotate:
    """Transformation class that applies a random rotation to a 3D pointcloud.

    Note that this class acts on pointclouds, whereas
    `torch_geometric.transforms.RandomRotate` acts on 3D meshes.

    Attributes:
        axes: a list or tuple of axes about which to perform rotations; for instance,
              `axes = ["z"]` means perform a rotation just around the z-axis, whereas
              `axes = ["x", "y", "z"]` means perform a SO(3) rotation. This is being
              passed directly to the scipy api, so order matters.
        fix_rotation: bool, apply a new rotation with every __call__ (false) or fix
                       the rotation once in advance (true).
    """

    def __init__(self, axes=("x", "y", "z"), fix_rotation=False):
        self.fix_rotation = fix_rotation
        # capitalization determines intrinsic vs extrinsic rotations in scipy
        # assume intrinsic rotations
        self.axes = "".join([ax.lower() for ax in axes])
        self.ax_to_idx = dict(x=0, y=1, z=2)
        if fix_rotation:
            self.set_transform()

    def set_transform(self):
        angles = np.random.uniform(0, 2 * np.pi, len(self.axes))
        self.angles = angles[0] if len(self.axes) == 1 else angles
        self.rotation = Rotation.from_euler(self.axes, self.angles)
        self.rotation_matrix = torch.from_numpy(self.rotation.as_matrix()).float()
        self.euler_angles = self.rotation.as_euler("xyz", degrees=True)

    def __call__(self, x):
        """Return a rotated version of `x`.

        Args:
            x: PyTorch Tensor with shape (N x 3) representing a 3D pointcloud. Note
               that we multiply by the transpose, but (N x 3) format is used for
               backward compatability with vector neuron experiments. We multiply by
               the transpose because 'These matrices produce the desired effect only
               if they are used to premultiply column vectors' according to wikipedia

        Returns:
            N x 3 tensor representing a rotated point cloud

        See Also:
            https://en.wikipedia.org/wiki/Rotation_matrix
        """
        if not self.fix_rotation:
            self.set_transform()

        return torch.matmul(x, self.rotation_matrix.T)

    @property
    def params(self):
        parameters = np.zeros(6)
        parameters[:3] = self.euler_angles
        return parameters

    @property
    def inv_params(self):
        return -self.params


class RandomTranslation:

    def __init__(self, means=None, stdevs=None, fix_translation=False):

        if not means:
            means = torch.ones(3)

        self.means = torch.tensor(means).float()

        if not stdevs:
            stdevs = torch.ones(3)

        self.stdevs = torch.tensor(stdevs).float()
        self.fix_translation = fix_translation

        if self.fix_translation:
            self.set_transform()

    def set_transform(self):
        self.translation = torch.normal(self.means, self.stdevs)

    def __call__(self, x):
        """Return a translated version of `x`.

        Args:
            x: N x 3 torch tensor

        Returns:
            A translated version of input
        """
        if not self.fix_translation:
            self.set_transform()

        return x + self.translation

    @property
    def params(self):
        parameters = np.zeros(6)
        parameters[3:] = self.translation.numpy()
        return parameters

    @property
    def inv_params(self):
        return -self.params


class RandomRigidBody:

    def __init__(
        self,
        axes=("x", "y", "z"),
        means=None,
        stdevs=None,
        fix_rotation=False,
        fix_translation=False,
    ):

        self.rotation_transform = RandomRotate(axes=axes, fix_rotation=fix_rotation)
        self.translation_transform = RandomTranslation(
            means=means, stdevs=stdevs, fix_translation=fix_translation
        )

    def __call__(self, x):

        x = self.rotation_transform(x)
        x = self.translation_transform(x)

        return x

    @property
    def params(self):
        parameters = np.zeros(6)
        parameters += self.translation_transform.params
        parameters += self.rotation_transform.params
        return parameters

    @property
    def inv_params(self):
        parameters = np.zeros(6)
        parameters += self.translation_transform.inv_params
        parameters += self.rotation_transform.inv_params
        return parameters


class PartitionRGBD:
    """A way of breaking up RGBD images for each sensor module.

    Later modules could receive partially overlapping pieces of images.
    """

    def __init__(
        self,
        x,
        y,
        x_img_size=320,
        y_img_size=240,
        retina_sensor="retina",
        depth_sensor="depth",
    ):
        """Assume we are receiving 240x320x3 img + 240x320 depth based on real_robots.

        Later can generalize this with image size parameters.

        TODO: just use reshape or view; this whole thing can be replaced with
        ~reshape(-1, x_patch, y_patch)
        in particular, view would avoid memory overhead / making copies
        """
        self.needs_rng = False

        self.x_dim = int(x)
        self.y_dim = int(y)

        self.x_img_size = x_img_size
        self.y_img_size = y_img_size

        self.retina_sensor = retina_sensor
        self.depth_sensor = depth_sensor

        msg = "Please make sure x_img_size and y_img_size are "
        msg += "divisible by x and y resp."
        assert self.x_img_size % self.x_dim == 0, msg
        assert self.y_img_size % self.y_dim == 0, msg

        self.n_steps_x = int(self.x_img_size / self.x_dim)
        self.n_steps_y = int(self.y_img_size / self.y_dim)

        self.x_grid = np.arange(0, self.x_img_size + self.x_dim, self.x_dim)
        self.y_grid = np.arange(0, self.y_img_size + self.y_dim, self.y_dim)

    def __call__(self, observation, state=None):
        # TODO: This transform needs be refactor in a way to include the agent_id
        # in addition to sensor_id and sensor type (semantic, depth, rgba).
        # https://github.com/thousandbrainsproject/tbp.monty/pull/57#discussion_r811267195
        retina, depth = (
            observation[self.retina_sensor],
            observation[self.depth_sensor],
        )
        img_patches = [[] for i in range(self.n_steps_y)]
        depth_patches = [[] for i in range(self.n_steps_y)]

        # Since you are stepping in the x-direction, x is over columns
        for x in range(self.n_steps_x):
            # Since you are stepping in the y-direction, y is over rows
            for y in range(self.n_steps_y):
                img_patch = retina[
                    self.y_grid[y] : self.y_grid[y + 1],
                    self.x_grid[x] : self.x_grid[x + 1],
                    :,
                ]

                depth_patch = depth[
                    self.y_grid[y] : self.y_grid[y + 1],
                    self.x_grid[x] : self.x_grid[x + 1],
                ]

                img_patches[y].append(img_patch)
                depth_patches[y].append(depth_patch)

        observation[self.retina_sensor] = img_patches
        observation[self.depth_sensor] = depth_patches

        return observation


class SelectRGBDPatch:
    """Select a patch from an RGBD image.

    Should not be used in conjunction with PartitionRGBD. For now assumes
    real_robots observation format - will adjust after merging habitat pr.
    """

    def __init__(
        self,
        row_start,
        row_end,
        col_start,
        col_end,
        retina_sensor="rgbd",
        depth_sensor="depth",
    ):
        self.needs_rng = False

        self.row_start = row_start
        self.row_end = row_end
        self.col_start = col_start
        self.col_end = col_end
        self.retina_sensor = retina_sensor
        self.depth_sensor = depth_sensor

    def __call__(self, observation, state=None):

        retina, depth = (
            observation[self.retina_sensor],
            observation[self.depth_sensor],
        )
        retina_patch = retina[
            self.col_start : self.col_end,
            self.row_start : self.row_end,
        ]
        depth_patch = depth[
            self.col_start : self.col_end,
            self.row_start : self.row_end,
        ]

        # Avoid pass by reference side effects since each sensor module is supposed
        # to be running its own transform
        new_observation = copy.deepcopy(observation)
        new_observation[self.retina_sensor] = retina_patch
        new_observation[self.depth_sensor] = depth_patch

        return new_observation

    def state_dict(self):
        return {
            "row_start": self.row_start,
            "row_end": self.row_end,
            "col_start": self.col_start,
            "col_end": self.col_end,
        }


class SelectPartitionedRGBDPatch:
    """Transform that picks out a single patch of an RGBD image.

    Used for testing out code before we have multiple sensor modules in play.

    Works downstream of PartitionRGBD.
    """

    def __init__(self, column, row, retina_sensor="retina", depth_sensor="depth"):
        self.needs_rng = False

        self.column = int(column)
        self.row = int(row)
        self.retina_sensor = retina_sensor
        self.depth_sensor = depth_sensor

    def __call__(self, observation, state=None):

        retina, depth = (
            observation[self.retina_sensor],
            observation[self.depth_sensor],
        )
        retina_patch = retina[self.column][self.row]
        depth_patch = depth[self.column][self.row]

        observation[self.retina_sensor] = retina_patch
        observation[self.depth_sensor] = depth_patch

        return observation
