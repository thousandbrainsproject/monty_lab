# Copyright 2025 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as rot  # noqa: N813
from sklearn.cluster import DBSCAN


def get_3d_transform_matrix(theta_tuple, cartesian_coord):
    """
    Return the full 4x4 transformation matrix for 3D space

    :param theta_tuple: The Euler-angles specifying orientation (in radians)
    :param cartesian_coord: (x,y,z) location
    """

    transform_matrix = np.identity(4)

    rot_component = rot.from_euler(
        "XYZ", theta_tuple  # Capitals specifies intrinsic/body reference frame
    )

    transform_matrix[0:3, -1] = cartesian_coord
    transform_matrix[0:3, 0:3] = rot_component.as_matrix()

    return transform_matrix


def linf_close(mat_a, mat_b, eta=0.00001):
    """
    Compute the L-inf distance between two matrices, and return true if the
    distance is small
    """

    distance = np.linalg.norm(x=abs(mat_a - mat_b), ord=np.inf)

    return distance <= eta


def normalize_translations(
    prediction_vector,
    minicolumn_dim,
):
    """
    Takes in the high-dimensional vector representing the full prediction of the
    HDCD LM, and normalizes the translation values to be bound between -1:1

    Note we later restore these after the consensus algorithm has converged

    :param prediction_vector: The prediction vector for normalization
    :param minicolumn_dim: The number of neurons per minicolumn

    TODO: remove this once phase-based encoding of space is implemented
    """

    max_displacement = np.max(abs(prediction_vector))
    # Should really only be looking at translation elements when
    # computing max, although in practice, these will almost always be larger
    # than rotation (which are bounded -1:1)

    for current_column in range(16):

        if (current_column + 1) % 4 == 0:  # Just consider location column

            index_from = current_column * minicolumn_dim
            index_to = (current_column + 1) * minicolumn_dim

            prediction_vector[:, index_from:index_to] = (
                prediction_vector[:, index_from:index_to] / max_displacement
            )

    return prediction_vector, max_displacement


def unnormalize_translations(
    prediction_vector,
    minicolumn_dim,
    max_displacement,
):
    """
    Reverses normalization, restoring the translation amounts to interpretable
    values; note that this is typically applied to the consenus prediction,
    which is therefore lower-dimensional than the array we applied normalization
    to

    :param prediction_vector: The prediction vector for normalization
    :param minicolumn_dim: The number of neurons per minicolumn
    :param max_displacement: The maximum value in the original predictions
    """

    for current_column in range(16):

        if (current_column + 1) % 4 == 0:

            index_from = current_column * minicolumn_dim
            index_to = (current_column + 1) * minicolumn_dim

            prediction_vector[index_from:index_to] = (
                prediction_vector[index_from:index_to] * max_displacement
            )

    return prediction_vector


def dynamic_routing(
    full_predictions,
    activated_neurons,
    exp_params,
):
    """
    Perform dynamic routing to find agreement in the estimated poses and object
    IDs. This is adapted from dynamic routing as used in Sabour et al, 2017,
    "Dynamic Routing Between Capsules".
    In short, begin by computing a uniformally weighted mean of all predictions;
    then, compare agreement between each prediction and this weighted mean, before
    then changing the weighting of each prediction based on its agreement (scalar
    product), and computing a new weighted mean. This process continues iteratively.

    :param full_predictions: The set of predictions (transformed inputs) for
        the LM, given in a sparse real-valued format
    :param activated_neurons: The neurons in the above set of predictions that
        are actually "spiking" (i.e. a binary array of appropriate ON values)
    :param exp_params: Dictionary of useful parameters for influencing the
        algorithm's performance
    """

    full_predictions, max_displacement = normalize_translations(
        full_predictions,
        exp_params["MINICOLUMN_DIM"],
    )

    # Take a weighted mean, initially assuming a uniform prior; TODO implement
    # learned priors over features (i.e. biases)
    num_predictions = full_predictions.shape[0]
    prediction_weights = np.ones(num_predictions) / num_predictions

    # Calculate the weighted mean
    weighted_consensus = np.sum(prediction_weights[:, None] * full_predictions, axis=0)

    for dr_iter in range(exp_params["DYNAMIC_R_ITERS"]):

        print(f"Dynamic routing iteration: {dr_iter}")

        # Calculate the scalar product (measure of agreement) between every
        # prediction and the current concensus
        broadcast_scalars = np.dot(weighted_consensus, full_predictions.T)

        # Generate soft-max based probability distribution
        prediction_weights = np.exp(
            exp_params["SOFTMAX_SCALING"] * broadcast_scalars
        ) / sum(np.exp(exp_params["SOFTMAX_SCALING"] * broadcast_scalars))

        """
        TODO make better use of the number of sensations; can give a warning
        if the majority of the distribution is not explained by k-weights
        when dynamic routing has completed, or indeed enforce this. Re.
        using this in the real world, when we won't know how many sensations
        of those we learned that we are actually experiencing (vs. here where
        we visit the same number), then this may not work. Instead, could have
        a heuristic that we continue until the explained cumm distribution
        is not changing signficiantly (up to a maximum number of dynamic
        routing iteratoins). If changing too slowly, could potentially even
        dyanmically increase the scaling constant. If scaling too fast (e.g.
        all of the weights are described by a single feature), then can
        decrease the scaling, and start over.
        """

        if exp_params["VISUALIZE_CUMM_DISTRIBUTION"]:
            # Visualize the cumulative distribution of prediction weights across
            # different iterations
            cum_sum_weights = np.cumsum(np.flip(np.sort(prediction_weights)))

            plt.plot(
                range(len(cum_sum_weights)),
                cum_sum_weights,
                linewidth=2,
                alpha=0.5,
                label="DR Iteration : " + str(dr_iter),
            )

        assert (
            abs(np.sum(prediction_weights) - 1) <= 0.001
        ), "Need probability distribution for weights"

        # Calculate the weighted consensus
        weighted_consensus = np.sum(
            prediction_weights[:, None] * full_predictions, axis=0
        )

    if exp_params["VISUALIZE_CUMM_DISTRIBUTION"]:
        # TODO save these to a directory with meaningful naming if these
        # are going to be used more
        plt.vlines(
            exp_params["NUM_F_FOR_SENSING"],
            0,
            1.0,
            alpha=0.7,
            color="k",
            label="Number of sensations",
        )  # We expect the "elbow"
        # of the cumm. distribution to be near the number of sensations,
        # assuming they are relatively familiar sensations for the LM
        plt.xlabel("Total predictions")
        plt.ylabel("Cumm. distribution of prediction weights")
        plt.legend()
        plt.show()
        plt.clf()

    # If using a hybrid algorithm with DBSCAN, return early the top
    # k predictions for further processing
    if exp_params["PROPORTION_FOR_EARLY_RETURN"] is not None:

        k = int(len(broadcast_scalars) * exp_params["PROPORTION_FOR_EARLY_RETURN"])
        # Can also consider using e.g. k = int(estimated_num_predictions * 10)
        top_k_indices = np.argpartition(broadcast_scalars, -k)[-k:]

        return (
            full_predictions[top_k_indices],
            activated_neurons[top_k_indices],
            max_displacement,
        )

    # Re-set the best prediction weights uniformally based on the top matches
    # This help prevents the weighted consensus from "over-fitting" on a single
    # prediction (which is going to be less robust than the mean)
    # TODO more naturally provide exp_params["NUM_F_FOR_SENSING"] from
    # the environment, accounting for potentially multiple input columns
    k = int(exp_params["NUM_F_FOR_SENSING"] * 0.5)
    top_k_indices = np.argpartition(broadcast_scalars, -k)[-k:]

    # Find the indices associated with the k top matches
    prediction_weights = np.zeros(full_predictions.shape[0])
    prediction_weights[top_k_indices] = 1 / len(top_k_indices)

    estimated_pose, winning_indices = extract_consensus_output(
        full_predictions,
        activated_neurons,
        prediction_weights,
        exp_params,
        max_displacement,
    )

    return estimated_pose, winning_indices


def dbscan_clustering(
    full_predictions, activated_neurons, exp_params, provided_max=None
):
    """
    Use the DBSCAN algorithm to find clusters in the high-dimensional, combined
    SDR-pose space; DBSCAN looks for clusters of points based on density;
    thus, it does not require specifying the number of clusters, and has the
    advantage that it can ignore outliers.

    Controlled by two hyper-parameters, one which is the window size for
    counting neighbors, and one which is the number of neighbors required
    in order to designate a point as a "core-point"; currently only the first
    can be set by the user in the run.py config; the number of neighbours
    is set as a heuristic based on the number of sensations

    Note that we currently assume there will be one cluster, i.e. there is one
    true object in the input; this will need to be revisited in the setting of
    multiple objects (e.g. cluttered scenes); e.g. rather than one, might
    have a heuristic that we are viewing <7 objects in a scene at any given
    time; NB this might also relate to human challenges around "crowding" and
    the serial nature of attention; eventually for multiple clusters, can use
    ?the number of points (relative to number of sensations) and variance
    (ideally should be low) to determine which is the best
    """

    if provided_max is None:
        # Normalize the translation displacements
        full_predictions, max_displacement = normalize_translations(
            full_predictions,
            exp_params["MINICOLUMN_DIM"],
        )
    else:
        # DBSCAN has been called as a follow-up to dynamic routing, so the
        # predictions are already normalized
        max_displacement = provided_max

    if exp_params["BINARY_SEARCH_BOOL"]:

        cluster_indices = binary_dbscan(
            full_predictions,
            exp_params,
            min_samples=int(exp_params["NUM_F_FOR_SENSING"] / 3),
        )

    else:

        hdcd_clusters = DBSCAN(
            eps=exp_params["WINDOW_SIZE"],
            min_samples=int(exp_params["NUM_F_FOR_SENSING"] / 3),
            metric="euclidean",
        ).fit(full_predictions)

        cluster_indices = hdcd_clusters.core_sample_indices_

    if (cluster_indices is None) or (len(cluster_indices) == 0):
        print("\nDBSCAN failed to find a cluster...\n")
        return None, None

    # Once the points of the cluster are identified, take a uniform, weighted
    # sum of these (i.e. the mean) as their consensus
    # While we could just calculate mean, we will re-use these
    # prediction weights for determining the output SDR
    prediction_weights = np.zeros(full_predictions.shape[0])
    prediction_weights[cluster_indices] = 1 / len(cluster_indices)
    # TODO can also later consider non-uniform sampling based on learned priors
    # about features and their relationship to certain objects

    assert (
        abs(np.sum(prediction_weights) - 1) <= 0.001
    ), "Prediction weighting should be a probability distribution"

    estimated_pose, winning_indices = extract_consensus_output(
        full_predictions,
        activated_neurons,
        prediction_weights,
        exp_params,
        max_displacement,
    )

    return estimated_pose, winning_indices


def binary_dbscan(
    predictions_vector,
    exp_params,
    min_samples,
):
    """
    Use a binary search to select the parameter values for DBSCAN (in particular
    the "epsilon" window size)

    min_samples : Estimate how densely packed we expect the
    sensations we experience to be; if learning only a subset of an
    object before performing inference, may want to make this smaller


    TODO in the future : also use the difference between the number of points in a
    cluster vs the expected number (where expected number is proportional to the
    number of sensations or the total number of input columns;  might give a more
    fine-grained metric for driving the search; e.g. this could be the
    absolute difference, and be used as a tie-break when num-clusters = 1 with
    both smaller and larger windows
    """

    print("\nBeginning binary search over DBSCAN parameters:")

    low_eps = 0
    high_eps = None

    print("Exponential search for starting points...")
    # Exponential search to find where we have more than one cluster
    epsilon = exp_params["WINDOW_SIZE"]
    for __ in range(exp_params["BINARY_ITERATIONS"]):
        hdcd_clusters = DBSCAN(
            eps=epsilon,
            min_samples=min_samples,
            metric="euclidean",
        ).fit(predictions_vector)

        labels = np.unique(
            hdcd_clusters.labels_,
        )

        if sum(labels > 0) >= 1:
            # We have multiple clusters, so this can be used as our upper
            # bound
            high_eps = epsilon
            print("Identified upper bound")
            print(f"high_eps: {high_eps}")
            break

        else:
            if sum(labels >= 0) == 1:  # One cluster

                print("Current eps gives 1 cluster; identified upper bound")
                high_eps = epsilon * 2
                print(f"high_eps: {high_eps}")
                break

            else:
                # This epsilon gives 0 clusters, so is an appropriate lower
                # bound
                low_eps = epsilon
                print("\nCurrent eps gives 0 clusters")
                print(f"low_eps: {low_eps}")

        epsilon *= 2

    if high_eps is None:
        print("Exponential search failed")
        return None

    # Binary search for optimal value
    for current_iteration in range(exp_params["BINARY_ITERATIONS"]):

        print(f"\nBinary-search, iteration : {current_iteration}")
        print(f"low_eps: {low_eps}")
        print(f"high_eps: {high_eps}")

        epsilon = (low_eps + high_eps) / 2.0

        hdcd_clusters = DBSCAN(
            eps=epsilon,
            min_samples=min_samples,
            metric="euclidean",
        ).fit(predictions_vector)

        labels = np.unique(
            hdcd_clusters.labels_,
        )

        if sum(labels >= 0) == 1:
            print("Converged to 1 cluster!")
            # TODO consider adding tie-break based on number of members of the
            # cluster; this should enable even tighter cluster estimates
            return hdcd_clusters.core_sample_indices_

        if sum(labels > 0) >= 1:
            high_eps = epsilon
            print("Window too large:")
            print(f"high_eps: {high_eps}")
        else:
            low_eps = epsilon
            print("Window too small:")
            print(f"low_eps: {low_eps}")

    return None


def extract_consensus_output(
    full_predictions,
    activated_neurons,
    prediction_weights,
    exp_params,
    max_displacement,
):
    """
    Given the consenus reached upstream for the prediction weights, determine
    the final outputs that will be returned by the learning module
    """
    # Re-calculate the weighted-consensus
    weighted_consensus = np.sum(prediction_weights[:, None] * full_predictions, axis=0)

    # Un-normalize the translation vectors
    weighted_consensus = unnormalize_translations(
        weighted_consensus,
        exp_params["MINICOLUMN_DIM"],
        max_displacement,
    )

    # Find the weighted *output neurons* (i.e. accounting for the fact
    # that some predictions may be for a 0-value pose, but might still indicate
    # the neuron with the highest probability of spiking)
    weighted_sdr_consensus = np.sum(
        prediction_weights[:, None] * activated_neurons, axis=0
    )

    # Iterate through each "mini-colum", selecting the winning neuron
    # TODO eventually consider using the k-winning neurons (i.e. >1)
    winning_indices = []
    for current_column in range(16):

        winning_indices.append(
            np.argmax(
                weighted_sdr_consensus[
                    current_column
                    * exp_params["MINICOLUMN_DIM"] : (
                        (current_column + 1) * exp_params["MINICOLUMN_DIM"]
                    )
                ]
            )
            + current_column * exp_params["MINICOLUMN_DIM"]
        )

    estimated_pose = np.reshape(weighted_consensus[winning_indices], (4, 4))

    return estimated_pose, winning_indices


def check_pose_estimate(
    actual_pose,
    predicted_pose,
    euler_eta,
    translation_eta,
):
    """
    Determine the correctness of the network's estimated pose.
    """

    print("\nChecking pose estimate...")

    print("GT pose:")
    print(actual_pose)

    # Convert the orientation matrices to Euler angle
    # NB that Scipy will automatically try to find the nearest orthonormal
    # matrix if the input does not satisfy this; TODO investigate if we want
    # to try alternative implementations of these algorithms
    actual_rot_scipy = rot.from_matrix(actual_pose[0:3, 0:3])
    predicted_rot_scipy = rot.from_matrix(predicted_pose[0:3, 0:3])

    assert linf_close(
        np.transpose(predicted_rot_scipy.as_matrix()) @ predicted_rot_scipy.as_matrix(),
        np.identity(3),
    ), "Predicted matrix is not orthogonal"

    euler_matched = np.all(
        abs(
            actual_rot_scipy.as_euler("zyx", degrees=False)
            - predicted_rot_scipy.as_euler("zyx", degrees=False)
        )
        <= euler_eta
    )

    # Compare Euclidean distances of the last columns of the full pose matrices
    euc_dist = np.linalg.norm(actual_pose[:, -1] - predicted_pose[:, -1])

    corrected_full_pose = np.zeros((4, 4))

    corrected_full_pose[0:3, 0:3] = predicted_rot_scipy.as_matrix()  # Use
    # orthonormal-enforced version

    corrected_full_pose[:, -1] = predicted_pose[:, -1]  # Recover translation

    print("Final predicted pose:")
    print(corrected_full_pose)

    print(f"\nRotation matched : {euler_matched}")

    euclidean_matched = euc_dist <= translation_eta
    print(f"Translation matched : {euclidean_matched}")

    return (euler_matched and euclidean_matched), corrected_full_pose


def check_classification(
    learned_dic, results_dic, target_id, inferred_sdr, sdr_classifier_threshold
):
    """
    Check if the output SDR corresponds to the actual object ID, or some other
    outcome

    learned_dic : The dictionary mapping between the SDRs learned by the LM,
    and the associated object ID

    Could speed this up by focusing checks on the target SDR - if sparsity is
    high, then may be able to quickly check that the output neurons aren't in
    the *union* of all other objects, in which case must be unique; TODO explore
    if this becomes slow with larger object data-sets
    """

    matched_object_ids = []

    print(
        f"\nChecking classification match with an SDR threshold of: "
        f"{sdr_classifier_threshold}..."
    )

    print("Target SDR:")
    print(learned_dic[target_id])
    print("Actual SDR:")
    print(inferred_sdr)

    # Iterate through all the learned objects
    for current_id, current_sdr in learned_dic.items():

        if len(np.intersect1d(current_sdr, inferred_sdr)) >= sdr_classifier_threshold:

            matched_object_ids.append(current_id)

    if len(matched_object_ids) > 1:
        print("Never converged, matched with multiple objects...")
        results_dic["wrong_converged"] += 1

    if len(matched_object_ids) == 0:
        print("No matched objects...")

        results_dic["no_match"] += 1

    if len(matched_object_ids) == 1:

        if matched_object_ids[0] == target_id:
            print("Matched to the correct object!")
            results_dic["correctly_converged"] += 1
        else:
            print("Converged to a single object, but the wrong object!")
            results_dic["wrong_converged"] += 1

    return results_dic


def plot_results(num_objects_list, base_dir):
    """
    Plot generated results
    """

    if os.path.exists(base_dir + "figures/") is False:
        try:
            os.mkdir(base_dir + "figures/")
        except OSError:
            pass

    font = {"size": 18}
    plt.rc("font", **font)

    with open(base_dir + "results.json", "r") as f:
        results_dic = json.load(f)

    # TODO add CI regions to the below line plots

    # ==Accuracy results==
    plt.plot(
        num_objects_list,
        results_dic["acc_vs_num_objects"],
        color="dodgerblue",
        alpha=0.7,
        linewidth=3,
    )
    plt.xlabel("Number of Objects in Environment")
    plt.ylabel("Classification Accuracy")
    plt.ylim(0, 1.05)
    plt.tight_layout()

    plt.savefig(base_dir + "figures/" + "acc_vs_num_objects.png", dpi=300)
    plt.clf()

    # ==Pose results==
    plt.plot(
        num_objects_list,
        results_dic["pose_recovery_vs_num_objects"],
        color="crimson",
        alpha=0.7,
    )
    plt.xlabel("Number of Objects in Environment")
    plt.ylabel("Pose-Recovery Success")
    plt.ylim(0, 1.05)
    plt.tight_layout()

    plt.savefig(base_dir + "figures/" + "pose_rec_vs_num_objects.png", dpi=300)
    plt.clf()

    # ==Computational time results==

    plt.plot(
        num_objects_list,
        results_dic["total_time_vs_num_objects"],
        color="rebeccapurple",
        alpha=0.7,
        label="Total time",
    )
    plt.plot(
        num_objects_list,
        results_dic["sense_time_vs_num_objects"],
        color="crimson",
        alpha=0.7,
        label="Sensations time",
    )
    plt.plot(
        num_objects_list,
        results_dic["pose_time_vs_num_objects"],
        color="dodgerblue",
        alpha=0.7,
        label="Pose time",
    )
    plt.xlabel("Number of Objects in Environment")
    plt.ylabel("Time for Inference")
    plt.legend()
    plt.tight_layout()

    plt.savefig(base_dir + "figures/" + "inference_time_vs_num_objects.png", dpi=300)
    plt.clf()
