# Copyright 2025 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


import numpy as np
import open3d as o3d
import torch
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from experiments.pose_estimation_utils import (
    best_fit_transform,
    matrix_to_params,
    nearest_neighbor_src_dst,
    params_to_matrix_rigid,
)

# TODO: handle initialization and other ways to improve ICP
# TODO: eventually want to be able to run some verison of this online
# NOTE: to run this script install open3d!


class Optimizer:
    """
    Simple class for sampling based optimizers like Langevin Sampling or MCMC
    """
    def __init__(self, **kwargs):

        self.n_steps = kwargs["n_steps"]
        self.current_step = 0
        self.param_history = []
        self.error_history = []
        self.step_history = []

    def reset(self):
        self.current_step = 0
        self.param_history = []
        self.error_history = []
        self.step_history = []

    def propose_params(self):
        """
        How the next parameters are sampled. Examples:
            In Metropolis Hastings, just add noise.
            In Langevin Sampling, do gradient decent + noise
        """
        pass

    def compute_error(self, params):
        """
        Takes in a set of parameters, computes error. Note that computing error could be
        complicated, e.g. might involve running a simulation, sampling a pointcloud from
        a mesh, etc.
        """
        pass

    def decision_fn(self):
        """
        EG in Metropolis Hastings, decide if you keep the proposed parameters or revert
        to the old ones. Should update current params and save stuff.
        """
        pass

    def step(self):
        """
        Propose new params, compute error, call decision fn to update params
        """
        pass

    def optimize(self):
        """
        Run a bunch of training steps
        """

        for _ in range(self.n_steps):
            self.step()

    def eval(self):  # noqa A003
        """
        To avoid a supervised experiment calling model.eval() and throwing error
        """
        pass

    def train(self):
        """
        To avoid a supervised experiment calling model.train() and throwing error
        """
        pass

    def best_params(self):
        """
        Used for extracting the best params at the end
        """
        pass


class PointCloudMetropolisHastings(Optimizer):
    """
    TODO: extend to rigid body transform. not hard to do, but not a high priority either
    so I might just triage this.
    """

    def __init__(self, translate=False, **kwargs):
        """
        Minimal MCMC sampler used for estimating rotation (and later translation)
        parameters.

        :param kappa: float, 1/variance for distribution to sample perturbations from
                      approaches uniform as k goes to zero from the right
        :param temp: float, std of normal dist for selecting new params or not
        """

        print("Warning, you are using some outdated code that is probably broken")

        super().__init__(**kwargs)

        self.temp = kwargs["temp"]
        self.kappa = kwargs["kappa"]
        threshold = kwargs.get("threshold", None)
        self.transforms = []

        if threshold:
            self.threshold = threshold

            def get_threshold():
                return self.threshold
        else:

            def get_threshold():
                return np.random.normal(0, self.temp)

        self.get_threshold = get_threshold

    def compute_error(self, params):
        """
        transform = transform_class(params), just translation + rotation
        new_points = transform(source_points)
        error = chamfer_distance(new_points, dest_points)
        """
        new_pc = self.apply_params(params, self.src)
        distances, _ = nearest_neighbor_src_dst(new_pc, self.dst)
        return distances.sum()

    def apply_params(self, params, pc):
        new_pc = Rotation.from_euler("xyz", params[:3]).apply(pc)
        new_pc += params[3:].reshape(1, 3)
        return new_pc

    def propose_params(self):
        """
        Since we are estimating angles (for now no translation), each new param needs
        to be on a periodic domain.
        """
        angle_noise = np.random.vonmises(0, self.kappa, 3)
        translation_noise = np.random.uniform(-10, 10, 3)
        new_params = np.zeros(6)
        new_params[:3] = self.current_params[:3] + angle_noise
        new_params[3:] = self.current_params[3:] + translation_noise
        return new_params

    def decision_fn(self):
        """
        Probably not remembering this right, double check
        """
        ratio = self.proposal_params_error - self.current_params_error
        ratio /= self.dst.shape[0]
        threshold = self.get_threshold()
        self.current_step += 1
        self.ratio_history[self.current_step] = ratio

        if ratio < threshold:
            self.current_params = self.proposal_params
            self.current_params_error = self.proposal_params_error
            self.step_history.append(self.current_step)
            self.error_history.append(self.current_params_error)
            self.param_history.append(self.current_params)

    def step(self):

        self.proposal_params = self.propose_params()
        self.proposal_params_error = self.compute_error(self.proposal_params)
        self.decision_fn()

    def __call__(self, src, dst):
        """
        :param dst: np.ndarray N_points x m_dimensions
        :param src: np.ndarray N_points x m_dimensions

        init points
        proposal points = perturb current points
        compute error of proposed points
        error_diff = compare error of proposed to existing
        T = draw random variable (forget exact dist)
        if error_diff < T: accept else reject
        """

        self.reset()
        # NOTE: for now assume batch size is 1 and squeeze batch dimension
        if len(src.size()) > 2:
            src = src.squeeze(dim=0)
            dst = dst.squeeze(dim=0)

        self.src = src
        self.dst = dst

        # Initialize params
        self.current_params = np.zeros(6)
        self.current_params[:3] = np.random.vonmises(0, 1, 3)
        self.current_params[3:] = np.random.uniform(-10, 10, 3)

        # Propose params
        self.proposal_params = self.propose_params()
        self.current_params_error = self.compute_error(self.current_params)
        self.proposal_params_error = self.compute_error(self.proposal_params)
        self.ratio_history = np.zeros(self.n_steps + 1)

        self.optimize()
        best_params = self.best_params
        best_pointcloud = self.apply_params(best_params, self.src)
        return torch.tensor(best_pointcloud).unsqueeze(dim=0)

    @property
    def best_params(self):
        """
        Metrics are computed in degrees, so switch over to degrees
        """
        if len(self.error_history) == 0:
            return self.current_params

        argmin = np.argmin(self.error_history)
        best_params = self.param_history[argmin]
        best_params[:3] = np.rad2deg(best_params[:3])
        return best_params

    def get_params(self, index):
        return self.param_history[index]


class IterativeClosestPoint(Optimizer):

    def __init__(self, **kwargs):
        """
        Based on https://github.com/ClayFlannigan/icp
        seealso
            https://gist.github.com/ecward/c373932638fd04a2243e
        """
        super().__init__(**kwargs)

    def step(self):

        distances, indices = nearest_neighbor_src_dst(
            self.src[:self.m, :].T, self.dst[:self.m, :].T
        )

        transform, _, _ = best_fit_transform(
            self.src[:self.m, :].T,
            self.dst[:self.m, indices].T
        )
        self.src = np.dot(transform, self.src)

        # log
        self.param_history.append(transform)
        self.error_history.append(np.sum(distances))
        self.current_step += 1

    def extract_transform(self, idx):

        transform = np.eye(self.m + 1)
        for i in range(idx + 1):
            transform = self.param_history[i].dot(transform)

        transform = matrix_to_params(transform)
        return transform

    @property
    def best_params(self):
        min_idx = np.argmin(self.error_history)
        return self.get_params(min_idx)

    def get_params(self, index):
        params = self.extract_transform(index)
        return params

    def __call__(self, src, dst):
        """
        Do an online optimization to match src to dst
        """

        self.reset()
        # NOTE: for now assume batch size is 1 and squeeze batch dimension
        if len(src.size()) > 2:
            src = src.squeeze(dim=0)
            dst = dst.squeeze(dim=0)

        self._src = src.T  # m x N
        self._dst = dst.T  # m x N
        self.m = self._dst.shape[0]

        # make points homogeneous, copy them to maintain the originals
        self.src = np.ones((self.m + 1, self._src.shape[1]))  # m + 1 x N
        self.dst = np.ones((self.m + 1, self._dst.shape[1]))  # m + 1 x N
        self.src[:self.m, :] = np.copy(self._src)
        self.dst[:self.m, :] = np.copy(self._dst)
        self.optimize()

        # reformat: remove homogenous coords, add back batch dim
        return torch.tensor(self.src[:self.m, :]).T.unsqueeze(dim=0)


class IterativeClosestPointScipyAlign(IterativeClosestPoint):

    def __call__(self, src, dst):

        self.reset()
        # NOTE: for now assume batch size is 1 and squeeze batch dimension
        if len(src.size()) > 2:
            src = src.squeeze(dim=0)
            dst = dst.squeeze(dim=0)

        self._src = src.T  # m x N
        self._dst = dst.T  # m x N
        self.m = self._dst.shape[0]

        # make points homogeneous, copy them to maintain the originals
        self.src = np.ones((self.m + 1, self._src.shape[1]))  # m + 1 x N
        self.dst = np.ones((self.m + 1, self._dst.shape[1]))  # m + 1 x N
        self.src[:self.m, :] = np.copy(self._src)
        self.dst[:self.m, :] = np.copy(self._dst)

        # subtract mean once in advance
        self.centroid_dst = np.mean(self.dst, axis=1).reshape(self.m + 1, 1)
        self.dst_centered = self.dst - self.centroid_dst

        # train and return best point cloud
        self.optimize()
        return torch.tensor(self.src[:self.m, :]).T.unsqueeze(dim=0)

    def step(self):

        distances, indices = nearest_neighbor_src_dst(
            self.src[:self.m, :].T, self.dst[:self.m, :].T
        )

        # Find rotation parameters with Kabsch algorithm, recover translation after
        self.centroid_src = np.mean(self.src, axis=1).reshape(self.m + 1, 1)
        self.src_centered = self.src - self.centroid_src
        rotation = Rotation.align_vectors(
            self.dst_centered[:self.m, indices].T,
            self.src_centered[:self.m, :].T)[0]
        translation = self.centroid_dst[:self.m].squeeze(1) - np.dot(
            rotation.as_matrix(),
            self.centroid_src[:self.m]).squeeze(1)

        # Update params, src
        params = np.zeros(6)
        params[:3] = rotation.as_euler("xyz", degrees=True)
        params[3:] = translation
        transform = params_to_matrix_rigid(params)
        self.src = np.dot(transform, self.src)

        # log
        self.param_history.append(transform)
        self.error_history.append(np.sum(distances))
        self.current_step += 1


class AlignVectors(Optimizer):

    def __call__(self, src, dst):

        self.reset()
        # NOTE: for now assume batch size is 1 and squeeze batch dimension
        if len(src.size()) > 2:
            src = src.squeeze(dim=0)
            dst = dst.squeeze(dim=0)

        self._src = src.T.numpy()  # m x N
        self._dst = dst.T.numpy()  # m x N
        self.m = self._dst.shape[0]

        # make points homogeneous, copy them to maintain the originals
        self.src = np.ones((self.m + 1, self._src.shape[1]))  # m + 1 x N
        self.dst = np.ones((self.m + 1, self._dst.shape[1]))  # m + 1 x N
        self.src[:self.m, :] = np.copy(self._src)
        self.dst[:self.m, :] = np.copy(self._dst)

        # Find centroids and subtract means
        self.centroid_dst = np.mean(self._dst, axis=1).reshape(3, 1)
        self.centroid_src = np.mean(self._src, axis=1).reshape(3, 1)
        self.dst_centered = self._dst - self.centroid_dst
        self.src_centered = self._src - self.centroid_src

        # Kabsch algorithm
        rotation = Rotation.align_vectors(self.dst_centered.T, self.src_centered.T)[0]
        translation = self.centroid_dst[:self.m].squeeze(1) - np.dot(
            rotation.as_matrix(),
            self.centroid_src).squeeze(1)

        # Best params are the only params
        self.best_params = np.zeros(6)
        self.best_params[:3] = rotation.as_euler("xyz", degrees=True)
        self.best_params[3:] = translation
        transform = params_to_matrix_rigid(self.best_params)
        output = np.dot(transform, self.src)

        return torch.tensor(output[:self.m, :]).T.unsqueeze(dim=0)


class AlignVectorsNoCentering(AlignVectors):

    def __call__(self, src, dst):

        self.reset()
        # NOTE: for now assume batch size is 1 and squeeze batch dimension
        if len(src.size()) > 2:
            src = src.squeeze(dim=0)
            dst = dst.squeeze(dim=0)

        self._src = src.T.numpy()  # m x N
        self._dst = dst.T.numpy()  # m x N
        self.m = self._dst.shape[0]

        # make points homogeneous, copy them to maintain the originals
        self.src = np.ones((self.m + 1, self._src.shape[1]))  # m + 1 x N
        self.dst = np.ones((self.m + 1, self._dst.shape[1]))  # m + 1 x N
        self.src[:self.m, :] = np.copy(self._src)
        self.dst[:self.m, :] = np.copy(self._dst)

        # Find centroids and subtract means
        self.centroid_dst = np.mean(self._dst, axis=1).reshape(3, 1)
        self.centroid_src = np.mean(self._src, axis=1).reshape(3, 1)
        self.dst_centered = self._dst - self.centroid_dst
        self.src_centered = self._src - self.centroid_src

        # Kabsch algorithm
        rotation = Rotation.align_vectors(self.dst.T, self.src.T)[0]
        translation = np.dot(rotation.as_matrix(), self.centroid_src).squeeze(1)

        # Best params are the only params
        self.best_params = np.zeros(6)
        self.best_params[:3] = rotation.as_euler("xyz", degrees=True)
        self.best_params[3:] = translation
        transform = params_to_matrix_rigid(self.best_params)
        output = np.dot(transform, self.src)

        return torch.tensor(output[:self.m, :]).T.unsqueeze(dim=0)


class IterativeClosestPointScipyMinimize(IterativeClosestPoint):

    def step(self):

        distances, indices = nearest_neighbor_src_dst(
            self.src[:self.m, :].T, self.dst[:self.m, :].T
        )

        result = minimize(
            pointcloud_distance_objective,
            np.zeros(6),
            args=(self.dst[:self.m, indices], self.src[:self.m, :]),
            method="Nelder-Mead"
        )
        transform = params_to_matrix_rigid(result.x)
        self.src = np.dot(transform, self.src)

        # log
        self.param_history.append(transform)
        self.error_history.append(np.sum(distances))
        self.current_step += 1


class IterativeClosestPointPoisson(IterativeClosestPoint):
    """
    At each time ICP iteration, use the scipy nelder mead minimization api to find
    parameters that minimize the distance of the src point cloud to the implicit
    function estimated by poisson surface reconstruction.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.poisson_args = kwargs["poisson_args"]

        print("Warning: IterativeClosestPointPoissont is not yet fully tested")
        print("Warning: it is also VERY slow, ~1.5 min per ICP iteration")

    def __call__(self, src, dst):
        """
        ICP replacing the minimization of one point cloud to nearest neighbors in
        another with distances given by an implicit function.
        """

        self.reset()
        # NOTE: for now assume batch size is 1 and squeeze batch dimension
        if len(src.size()) > 2:
            src = src.squeeze(dim=0)

        self.dst = dst
        self.src = src.T  # m x N

        # Rebuild the mesh via Poisson surface reconstruction
        self.reconstruct()
        # Minimize distance of src to implicit surface
        self.optimize()

    def reconstruct(self):
        self.mesh = points_to_poisson_surface(self.dst, **self.poisson_args)

    def step(self):

        result = minimize(
            distance_to_implicit_objective,
            np.zeros(6),
            args=(self.mesh, self.src),
            method="Nelder-Mead"
        )

        transform = params_to_matrix_rigid(result.x)
        self.src = np.dot(transform, self.src)
        error = distance_to_implicit_objective(result.x, self.mesh, self.src)

        self.param_history.append(transform)
        self.error_history.append(np.sum(error))
        self.current_step += 1


def pointcloud_distance_objective(parameters, dst, src):
    """
    Let's just see if this api works for rotation only for now

    TODO: verify results are same as params_to_matrix(params).dot(src)
    """

    euler_angles = parameters[:3]
    translation = parameters[3:]
    r = Rotation.from_euler("xyz", euler_angles, degrees=True)
    est = r.apply(src.T).T + translation.reshape(3, 1)
    error = np.linalg.norm(est - dst, axis=0).sum()

    return error


def query_dist_to_surface(scene, query):
    """
    Given an open3d scene (output of mesh_to_implicit) and a query point ([,3] tensor)
    measure 3 types of distances and return in a single array.
    """
    typed_query = o3d.core.Tensor([query], dtype=o3d.core.Dtype.Float32)
    signed_distance = scene.compute_signed_distance(typed_query)
    return signed_distance


def mesh_to_implicit(mesh):
    """
    Take an open3d triangle mesh and create a ray casting scene from whih we can compute
    distances of query points.
    """
    mesh_ = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_)
    return scene


def distance_to_implicit_objective(parameters, reconstructed_mesh, src):

    euler_angles = parameters[:3]
    translation = parameters[3:]
    r = Rotation.from_euler("xyz", euler_angles, degrees=True)
    est = r.apply(src) + translation.reshape(1, 3)

    n_points = est.shape[0]
    estimated_distances = np.zeros(n_points)
    scene = mesh_to_implicit(reconstructed_mesh)
    for i in range(n_points):
        query = est[i, :]
        estimated_distances[i] = query_dist_to_surface(scene, query).numpy()

    return np.sum(estimated_distances ** 2)


def points_to_poisson_surface(obj, depth=9, **kwargs):
    """
    Take in a torch tensor and
        1) convert to o3d point cloud
        2) estimate point normals
        3) do poisson surface reconstruction

    :param obj: tensor[n x 3]
    :param depth: int how deep the octree; default 9 based on tutorial
    :param kwargs: other stuff passed to create_from_point_cloud_poisson
    :return mesh: o3d triangle mesh reconstruction
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obj)
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # clear existing data
    pcd.estimate_normals()
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as _:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=depth,
            **kwargs)

    return mesh
