# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import copy
import math
import random
import time

import numpy as np

import utils


class HighDimCoincidenceEnvironment:
    """
    Simple environment to enable creating and e.g. feeding in a variety of
    synthetically generated objects
    Useful for testing functionality of HDCD learning modules
    """

    def __init__(
        self,
        minicolumn_dim,
        num_objects=1,
        num_f_per_object=1,
        preset_feature_indices=None,
    ):

        self.minicolumn_dim = minicolumn_dim  # Note that because we assume objects
        # feature's will already be in the required SDR format for the learning
        # module, we require the minicolumn_dim when creating them
        self.num_objects = num_objects
        self.num_f_per_object = num_f_per_object
        self.preset_feature_indices = preset_feature_indices

        self.objects_list = []
        # Populate the environment with synthetic objects
        for object_id in range(self.num_objects):

            print(f"Creating synthetic object # : {object_id}")

            self.objects_list.append(
                HighDimCoincidenceObject(
                    self.minicolumn_dim,
                    object_id,
                    num_features=self.num_f_per_object,
                    preset_feature_indices=self.preset_feature_indices,
                )
            )

        self.object_iter = 0
        self.current_object = self.objects_list[self.object_iter]

        self.sensation_iter = 0

    def reset_object_iter(self):
        """
        Move back to the first object, e.g. before beginning inference
        """

        self.object_iter = 0
        self.current_object = self.objects_list[self.object_iter]

    def next_object(self):

        self.object_iter += 1
        try:
            self.current_object = self.objects_list[self.object_iter]
        except IndexError:
            print("\nNo more objects available to use. Returning to first object.")
            self.object_iter = 0
            self.current_object = self.objects_list[self.object_iter]

        self.reset_sensations()

    def current_sensation(self):
        """
        In an ordered sequence, iterate over the features associated with
        an object; eventually would implement ability to follow some more
        arbitrary sequence, but not v. significant for synthetic objects
        """

        current_feature_dic = self.current_object.return_feature_properties(
            self.sensation_iter
        )

        self.sensation_iter += 1

        return current_feature_dic

    def reset_sensations(self):
        """
        Reset sensations, e.g. at the start of inference, but after learning
        has taken place, or before performing inference on another object
        """
        self.sensation_iter = 0


class HighDimCoincidenceObject:
    """
    Simple synthetic objects used for evaluating the functionality of
    HDCD learning modules. For convenience, we assume that the features on these
    objects already have the phasic-SDR format that we will need at inference time,
    although in reality, this conversion would need to be done by e.g. a sensory module.
    """

    def __init__(
        self, minicolumn_dim, object_id, num_features=1, preset_feature_indices=None
    ):
        self.minicolumn_dim = minicolumn_dim
        self.object_id = object_id
        self.num_features = num_features

        self.object_orientation = np.identity(n=4)
        # The orientation of the entire object initialized to the identity

        # Store the values (i.e. pose directions) and fire_probs (essentially the
        # estimated significance of a neuron) separately; these lists will be
        # the features associated with different sensations on an object
        self.features_list_vals = []
        self.features_list_fire_probs = []
        self.features_list_dense_poses = []
        self.features_list_original_poses = []  # Store the original (i.e. at
        # object creation) pose of each feature relative to the object; makes
        # it easier later to work-out the pose of each feature after a series of
        # object transformations; both original poses and dense poses will be lists
        # of length=num_features; each dense_pose element is a 4x4 transformation matrix
        # whereas each original_pose element is a list of the Euler-angles and xyz
        # position of the feature

        for __ in range(self.num_features):
            """
            Currently initilaizes each feature to a random pose/orientation
            relative to the object; these will be placed (at the time of object
            creation) somewhere in a cube around the centre of the object
            """

            self.generate_feature(
                theta_tuple=np.random.uniform(0, 1, size=3) * (2 * math.pi),
                cartesian_coord=[
                    np.random.uniform(-5, 5),
                    np.random.uniform(-5, 5),
                    np.random.uniform(-5, 5),
                ],
                preset_feature_indices=preset_feature_indices,
            )

    def get_object_pose(self):
        """
        Return the object's actual pose (which will always be in the "dense"
        format, unlike feature poses)
        """
        return self.object_orientation

    def generate_feature(
        self, theta_tuple, cartesian_coord, preset_feature_indices=None
    ):
        """
        For what will correspond to a single sensation, generate the
        corresponding feature that is found on the object

        This is a similar process to asigning minicolumns during learning,
        i.e. there should be one neuron per minicolumn that is "on" (i.e. have
        non-zero fire_prob), and this neuron's value should encode an element
        of the full pose matrix

        Unlike before, the pose will not necessarily be the identity, and
        instead is determined by the input parameters

        If preset_feature_indices is not None, it should specify a list (e.g. 5
        different features), where each "feature" is the indices associated with
        each minicolumn; this evaluates performance when SDRs are re-used,
        rather than largely unique and independent for every feature
        """

        # The feature arrays will eventually correspond to the full dimension of
        # each column (here a sensory rather than learning module); vals
        # correspond to the value in the pose matrix, fire_probs to probability
        # that a particular neuron fires (i.e. the binary SDR representation),
        # rather than *when* it fires (the value)
        feature_vals = [[] for i in range(4)]
        feature_fire_probs = [[] for i in range(4)]

        dense_feature_pose = utils.get_3d_transform_matrix(theta_tuple, cartesian_coord)

        # If using a preset feature, first select (randomly) which of these
        # preset features we will be using
        if preset_feature_indices is not None:
            chosen_feature = preset_feature_indices[
                np.random.choice(len(preset_feature_indices))
            ]

        neuron_gen_index = 0  # Keep track of which value from
        # preset_feature_indices to use (assuming it's not None)

        for row_iter in range(4):

            for column_iter in range(4):

                # Create lists of 0s for this minicolumn
                local_vals = [0] * self.minicolumn_dim
                local_fire_probs = [0] * self.minicolumn_dim

                # Select the "on" neuron
                if preset_feature_indices is not None:
                    # Note that when generating a feature, we consistently
                    # index into the chosen feature, based on which minicolumn
                    # we are concerned with
                    neuron_id = chosen_feature[neuron_gen_index]
                else:
                    neuron_id = random.randint(0, self.minicolumn_dim - 1)

                local_vals[neuron_id] = dense_feature_pose[row_iter, column_iter]

                local_fire_probs[neuron_id] = 1.0

                feature_vals[row_iter].append(local_vals)
                feature_fire_probs[row_iter].append(local_fire_probs)

                neuron_gen_index += 1

        assert np.sum(feature_fire_probs) == 16, "Too many ON values in new feature"
        # Because each minicolumn can only have one winning neuron, and there are 4x4
        # minicolumns corresponding to the transformation matrix, this should always be
        # 16

        self.features_list_vals.append(np.array(feature_vals))
        self.features_list_fire_probs.append(np.array(feature_fire_probs))

        self.features_list_dense_poses.append(dense_feature_pose)
        self.features_list_original_poses.append((theta_tuple, cartesian_coord))

    def return_feature_properties(self, sensation_iter):
        """
        Return the values, fire_probs, and dense pose matrix for the current
        feature
        """

        current_feature_dic = dict(
            vals=self.features_list_vals[sensation_iter],
            fire_probs=self.features_list_fire_probs[sensation_iter],
            dense_pose=self.features_list_dense_poses[sensation_iter],
        )

        return current_feature_dic

    def transform_object(
        self, theta_tuple, cartesian_coord, theta_noise=0, cartesian_noise=0
    ):
        """
        Transform (e.g. rotate and shift) an entire object

        If the noise parameters are floats >0, each feature's transformation will
        be perturbed by a small amount (i.e. beyond the global transformation),
        simulating noise in the relative position of local features, e.g. a new
        example of a particular object, or sampling from new points on a
        familiar object
        """

        transformation_m = utils.get_3d_transform_matrix(theta_tuple, cartesian_coord)

        print("Original object orientation:")
        print(self.object_orientation)
        self.object_orientation = self.object_orientation @ transformation_m
        print("New object orientation:")
        print(self.object_orientation)

        for feature_iter in range(self.num_features):

            original_theta = self.features_list_original_poses[feature_iter][0]
            original_cartesian_coords = self.features_list_original_poses[feature_iter][
                1
            ]

            perturbed_theta = original_theta + np.random.normal(0, theta_noise)

            perturbed_cartesian_coord = original_cartesian_coords + np.random.normal(
                0, cartesian_noise, size=len(cartesian_coord)
            )

            perturbed_object_centric_pose = utils.get_3d_transform_matrix(
                perturbed_theta, perturbed_cartesian_coord
            )

            self.transform_feature(
                self.object_orientation, perturbed_object_centric_pose, feature_iter
            )

    def transform_feature(
        self, transformation_m, features_object_centric_pose, feature_iter
    ):
        """
        Perform a transformation on a feature (i.e. in the global evironment,
        here equivalent to body-centric coordinates), based on the transformation
        of its parent object, as well as any additional noise that is desired to
        make the task of recognition more challenging

        This updates the "live" pose of each feature, i.e. relative to the
        environment, however the "original" pose (relative to the object) is left
        untouched

        NB this will need refactoring to work if there are multiple ON neurons
        for each input minicolumn; currently this is always just 1
        """

        # First compute the change to the dense representation; note that the pose
        # contains both the orientation and the location in space
        self.features_list_dense_poses[feature_iter] = (
            transformation_m @ features_object_centric_pose
        )

        # Then for each of the entries in the dense representation, find the
        # appropriate mini-column neuron that has a non-zero fire_prob, and update
        # its pose
        for row_iter in range(4):

            for column_iter in range(4):

                mask = np.where(
                    self.features_list_fire_probs[feature_iter][row_iter][column_iter]
                    > 0
                )

                # Set the ON values to be equivalent to the result following
                # the transformation of the dense matrix
                self.features_list_vals[feature_iter][row_iter][column_iter][
                    mask
                ] = self.features_list_dense_poses[feature_iter][row_iter][column_iter]


class HighDimCoincidenceLM:
    def __init__(self, minicolumn_dim):
        """
        Learning Module that uses neurons performing high dimensional
        coincidence detection. In particular, uses learned transformations
        to predict the pose of an object and it's identity jointly using an
        encoding that combines SDR representations with the pose of an object.
        """

        self.mode = "pretraining"
        self.minicolumn_dim = minicolumn_dim

        self.reset_lm_predictions()  # Initialize data structures for holding
        # the current representation (pose and ID) of the object

        # Lists to hold the learned weights that will transform
        # inputs at inference time
        self.all_transformation_weights = []
        self.all_alignment_weights = []
        self.output_routing = []  # This will store which neurons in each minicolumn
        # should receive incoming signals, and therefore encode pose, following a
        # particular transformation of an input sensation; this is stored as a list
        # of binary arrays, where "1" indicates that a neuron should have information
        # routed to it, 0 otherwise

    def reset_lm_predictions(self):
        """
        Reset the predictions and current output consensus of the LM

        NB that in the future, we may want to e.g. maintain a particular
        pose or begin with particular hypothesis e.g. upright identity
        is more probable than many random orientations, and actually have this
        influence inference
        """

        self.dense_pose = np.identity(4)  # Note that this doesn't influence inference;
        # it just means that if we query the pose of an initialized LM, it will return
        # the identity pose
        self.consensus_sdr = []

        # Data structures for accumulating predictions
        self.inference_activated_neurons = []  # Neurons that have received some
        # input during inference; each member of the list is represented by
        # a sparse binary array
        self.predictions_points = []  # A list of the full prediction
        # representation, i.e. a sparse array with indices corresponding to the
        # SDR, and values to the predicted pose

    def set_lm_for_learning_object(self):
        """
        Randomly select the output neurons in each minicolumn that will be used
        for learning a new object; eventually, would want there to be some
        guiding heuristic or learning signal such that similar objects have
        similar SDRs assigned for their learning
        """
        # As elsewhere, the value of 16 is hardcoded due to the 4x4 size of
        # transformation matrices
        for current_column in range(16):

            self.consensus_sdr.append(
                (
                    random.randint(0, self.minicolumn_dim - 1)
                    + (current_column * self.minicolumn_dim)
                )
            )

        print("Output indices selected for learning on new object:")
        print(self.consensus_sdr)

    def get_dense_pose_and_sdr(self, clustering_method=None, exp_params=None):
        """
        Return the current best proposed pose of the object, as well as the
        estimated SDR. Currently, we do not consider returning multiple possible
        object IDs and poses, although this could be cosidered in the future.

        Options for clustering methods are:
        : DBSCAN
        : dynamic_routing
        : hybrid_clustering : dynamic routing to reduce the number of prediction
        points, then DBSCAN for refinement
        """

        if len(self.predictions_points) > 0:
            # If any predictions have been made and we want the pose and
            # SDR, then perform clustering to determine the LM's current
            # consensus"

            print(f"\nDetermining consensus with : {clustering_method}")

            if clustering_method == "dbscan":

                inferred_pose, inferred_sdr = utils.dbscan_clustering(
                    full_predictions=np.array(self.predictions_points),
                    activated_neurons=np.array(self.inference_activated_neurons),
                    exp_params=exp_params,
                )

            elif clustering_method == "dynamic_routing":

                assert (
                    exp_params["PROPORTION_FOR_EARLY_RETURN"] is None
                ), "Only want early return for hybrid clustering"

                inferred_pose, inferred_sdr = utils.dynamic_routing(
                    full_predictions=np.array(self.predictions_points),
                    activated_neurons=np.array(self.inference_activated_neurons),
                    exp_params=exp_params,
                )

            elif clustering_method == "hybrid_clustering":

                assert (
                    exp_params["PROPORTION_FOR_EARLY_RETURN"] is not None
                ), "Please specify the proportion of predictions to return"

                print("\n\nUsing hybrid clustering:")
                print("Size of predictions before first pruning:")
                print(np.shape(self.predictions_points))
                # Note the additional returned max_disp when doing the hybrid version
                (
                    pruned_predictions,
                    pruned_activations,
                    max_disp,
                ) = utils.dynamic_routing(
                    full_predictions=np.array(self.predictions_points),
                    activated_neurons=np.array(self.inference_activated_neurons),
                    exp_params=exp_params,
                )
                print("Size of predictions after pruning by dynamic-routing:")
                print(np.shape(pruned_predictions))

                inferred_pose, inferred_sdr = utils.dbscan_clustering(
                    full_predictions=pruned_predictions,
                    activated_neurons=pruned_activations,
                    exp_params=exp_params,
                    provided_max=max_disp,
                )

            else:
                raise ValueError("Please specify a valid clustering method")

            self.dense_pose = inferred_pose
            self.consensus_sdr = inferred_sdr

        else:
            # Outside of inference e.g. during learning, simply return
            # the current pose set in the LM
            pass

        return self.dense_pose, self.consensus_sdr

    def set_mode(self, input_mode):

        print(f"\nChanging learning module mode to : {input_mode}")

        self.mode = input_mode

    def process_sensation(
        self,
        current_feature_dic,
        learning_sdr=None,
        alignment_threshold=None,
    ):
        """
        learning_sdr should be provided when performing learning; this
        represents the output SDR for the current object, and therefore
        how the outputs of the learned weight transformation will route
        their values to the receiving column/LM
        """
        assert (
            self.mode != "pretraining"
        ), "Do not feed sensations before training or inference has begun."

        if self.mode == "training":

            self.learn_weight_set(current_feature_dic, learning_sdr)

        if self.mode == "inference":

            assert (
                learning_sdr is None
            ), "Don't provide learning signals at inference"

            self.predict_pose_and_id(current_feature_dic, alignment_threshold)

    def learn_weight_set(self, current_feature_dic, learning_sdr):
        """
        Perform learning by generating weight transformation matrices
        that correspond to the pose and SDR of the input, and the relation
        of this pose-SDR to the pose-SDR of the output (where the pose of
        the output at learning is assumed to be the identity matrix for
        mathematical simplicity)
        """

        weights_to_learn = np.identity(4)

        # Determine the inverse matrix of the input; this will define
        # the weight values for learning
        rotation_transpose = (current_feature_dic["dense_pose"][:3, :3]).T

        original_translation = current_feature_dic["dense_pose"][:3, -1]

        weights_to_learn[:3, :3] = rotation_transpose

        # Set translation column of the inverse matrix
        weights_to_learn[:3, -1] = (-1 * rotation_transpose) @ original_translation

        assert utils.linf_close(
            current_feature_dic["dense_pose"] @ weights_to_learn, np.identity(4)
        ), "Dense weights for learning are not inverse"

        # We are going to perform multiple different weight operations
        # (e.g. dendritic segments) on each row of the input, so repeat
        # the rows
        feature_vals = np.reshape(
            current_feature_dic["vals"], (4, 4 * self.minicolumn_dim)
        )
        broadcast_feature_vals = np.repeat(feature_vals, repeats=4, axis=0)

        # The indices of the input associated with firing determine
        # the connectivity to the input SDR
        input_prob_mat = np.reshape(
            current_feature_dic["fire_probs"], (4, 4 * self.minicolumn_dim)
        )

        # Build the sparse weight operation associated with this feature
        # Note that while this process might be vectorized, it is only
        # performed at learning, and only once per input feature, so this
        # is currently sufficiently fast; also note that, once again,
        # as we are going to be re-using weights (*but with different
        # associated indices*), we build up a larger array than what would be used
        # for a standard matrix product
        combined_transform_weights = None

        for row_index in range(4):
            indices = np.where(input_prob_mat[row_index, :] > 0)
            temp_weight_matrix = np.zeros((4 * self.minicolumn_dim, 4))

            for column_index in range(4):

                temp_weight_matrix[indices, column_index] = weights_to_learn[
                    :, column_index
                ]
                # Asign the weights to the appropriate indices of he
                # weight matrix

            if combined_transform_weights is None:
                combined_transform_weights = copy.copy(temp_weight_matrix)
            else:
                combined_transform_weights = np.concatenate(
                    (combined_transform_weights, temp_weight_matrix), axis=1
                )

        # Alignment weights are simpler, because they can take on only a
        # single value (1), and so do not need to be repeated
        alignment_weights = np.zeros((4 * self.minicolumn_dim, 4))
        alignment_weights.T[np.where(input_prob_mat > 0)] = 1

        """
        Check the transform aand alignment weights we've just made

        Perform a series of "element-wise" dot-products between two lists
        of vectors; this can be efficiently expressed and performed using
        Einstein summation, and will be extended for inference, so we
        introduce the simpler form here

        Explanation of the below Einstein summation:
        - ij,ij indicates that we are going to multiply the j-indexed
        elements of each ith row together (i.e. multiply/dot product each of
        the ith rows), and then sum over the j values; summation is indicated
        by "->i" i.e. omitting j in the output, which indicates that
        we don't sum across the i rows, but we do for everything else

        See : https://ajcr.net/Basic-guide-to-einsum/
        as well as :
        https://techtalk.digitalpress.blog/decoding-einsum-with-python/
        for useful guides
        """
        identity_check_two = np.reshape(
            np.einsum("ij,ij->i", broadcast_feature_vals, combined_transform_weights.T),
            (4, 4),
        )

        assert utils.linf_close(
            identity_check_two, np.identity(4)
        ), "Sparse transform weights for learning are not inverse"

        # Perform an elementwise multiplication and then sum across all
        # elements
        alignment_check = np.einsum("ij,ij->", input_prob_mat, alignment_weights.T)
        assert alignment_check == 16, "Error in alignment weights"

        # Combine the new weights with those previously learned by the LM
        self.all_transformation_weights.extend(combined_transform_weights[:, :, None].T)
        self.all_alignment_weights.extend(alignment_weights[:, :, None].T)

        # Save the array of output neurons associated with this transformation
        activated_outputs = np.zeros(16 * self.minicolumn_dim)
        activated_outputs[learning_sdr] = 1
        self.output_routing.extend(activated_outputs[:, None].T)

    def weights_to_nparray(self):
        """
        After having assembled all the weights associated with learning,
        convert to numpy representations for later inference

        We accumulated with list.append() due to its favourable computational
        efficiency
        """

        self.all_transformation_weights = np.array(self.all_transformation_weights)
        self.all_alignment_weights = np.array(self.all_alignment_weights)
        self.output_routing = np.array(self.output_routing)

    def predict_pose_and_id(
        self, current_feature_dic, alignment_threshold, debugging_mode=False
    ):
        """
        Given the input feature and learned weights, predict the output
        SDR and pose of the object; at this point, we will make many different
        predictions based on all of these weights; later, we will look
        for a consensus in these predictions to actually come to an agreed
        object ID and pose; this is similar to what is performed with Hough
        transforms or Capsule networks.

        This will involve performing the same operations as we did during
        learning to check the correctness of our learned weights, but in a
        vectorized form across all of the learned weight operations

        debugging_mode : if True (needs to be set manually), run a slow for
        loop to check the output results
        """
        feature_vals = np.reshape(
            current_feature_dic["vals"], (4, 4 * self.minicolumn_dim)
        )
        fire_probs = np.reshape(
            current_feature_dic["fire_probs"], (4, 4 * self.minicolumn_dim)
        )

        # First leverage alignment weights, checking which weight sets we
        # care about based on alignment in SDR space between the feature and the
        # learned transformation
        allignment_vec_results = np.einsum(
            "ij,kij->k",
            fire_probs,
            self.all_alignment_weights,
        )

        # Then based on the above result, sub-select the relevant
        # transformations for the full operation, reducing the number of
        # computations
        aligned_indices = np.where(allignment_vec_results >= alignment_threshold)[0]

        broadcast_feature_vals = np.repeat(feature_vals, repeats=4, axis=0)

        # We use Einstein summation again to check every learned (and
        # sufficiently aligned) weight transformation's prediction of the
        # object pose.
        # For each of the k different, sufficiently aligned weight
        # transformations, we are going to compute the predicted pose by
        # multiplying the i rows, and within them, the j values corresponding
        # to neurons/actual weights, then summing over these j values.
        # k is the additional dimension corresponding to all the different
        # weight sets
        pose_predictions = np.einsum(
            "ij,kij->ki",
            broadcast_feature_vals,
            self.all_transformation_weights[aligned_indices, :, :],
        )  # The pose predictions, output as a dense representation

        # Determine the different predictions for the binary array of the
        # output (i.e. the SDR of the object-level)
        new_activations = self.output_routing[aligned_indices, :]
        mask = np.ma.make_mask(new_activations)

        if debugging_mode:
            for current_prediction in range(len(aligned_indices)):
                assert (
                    len(np.where(new_activations[current_prediction] > 0)[0]) == 16
                ), "Incorrect number of output neurons selected for routing"

        # Asign the pose predictions to sparse arrays, indexed based on
        # new_activations
        new_predictions = np.zeros((len(aligned_indices), 16 * self.minicolumn_dim))

        new_predictions[mask] = pose_predictions[:, :].flatten()

        self.inference_activated_neurons.extend(new_activations)
        self.predictions_points.extend(new_predictions)


def simulate_hdcd_module_and_inference(
    exp_params, seed, num_objects, preset_feature_indices=None
):
    """
    The main simulation loop over synthetic objects, performing
    i) object creation
    ii) LM learning of the objects
    iii) (optional) transformation of the objects after learning
    iv) inference
    """

    random.seed(seed)
    np.random.seed(seed)

    print("\n\n\n===STARTING MAIN EXPERIMENT===")

    print("\nInitializing learning environment...")
    simple_environment = HighDimCoincidenceEnvironment(
        minicolumn_dim=exp_params["MINICOLUMN_DIM"],
        num_objects=num_objects,
        num_f_per_object=exp_params["NUM_F_PER_OBJECT"],
        preset_feature_indices=preset_feature_indices,
    )

    print("\nInitializing learning module...")
    hdcd_lm = HighDimCoincidenceLM(minicolumn_dim=exp_params["MINICOLUMN_DIM"])

    print("\n\n\n===BEGGINING TRAINING===")

    hdcd_lm.set_mode("training")

    learned_objects_dic = {}  # Dictionary of learned object associating their
    # object ID with the SDR present in the output layer; used for checking
    # classification accuracy

    # ***Iterate through the objects for learning***
    for __ in range(num_objects):
        print(
            "\nCurrently LEARNING on object: "
            + str(simple_environment.current_object.object_id)
        )

        hdcd_lm.reset_lm_predictions()
        hdcd_lm.set_lm_for_learning_object()

        _, current_sdr = hdcd_lm.get_dense_pose_and_sdr()
        # NB the SDR identified at learning becomes the "label"
        # for checking classification at inference

        learned_objects_dic[simple_environment.current_object.object_id] = current_sdr

        # Feed the learning module the next sensation of the object; note that
        # we currently learn on all features of an object
        for __ in range(exp_params["NUM_F_PER_OBJECT"]):

            # NB current_sensation() will automatically finish by ensuring that
            # on the next sensation, we will point to the next feature
            hdcd_lm.process_sensation(
                simple_environment.current_sensation(),
                learning_sdr=current_sdr,
            )

        # Get the next object; NB this automatically resets the sensation iter
        simple_environment.next_object()

    hdcd_lm.weights_to_nparray()  # Update the representation of the combined
    # weights created during learning

    # *** After training, induce random, rigid-body rotations to each object ***
    print("\n\n===PERFORMING RANDOM RIGID OBJECT TRANSFORMATIONS===")

    simple_environment.reset_object_iter()

    for __ in range(num_objects):
        print(f"\nTransforming object: {simple_environment.current_object.object_id}")

        random_obj_rotation = exp_params["RANDOM_ROTATIONS_MAG"] * np.random.uniform(
            0, 1, size=3
        )
        print(f"Rotating by: {random_obj_rotation}")

        random_obj_translation = exp_params[
            "RANDOM_TRANSLATIONS_MAG"
        ] * np.random.uniform(-1, 1, size=3)
        print("Translating by:")
        print(random_obj_translation)

        # Transform the objects from their learned, identity/canonical pose
        simple_environment.current_object.transform_object(
            random_obj_rotation,
            random_obj_translation,
            theta_noise=exp_params["THETA_NOISE"],
            cartesian_noise=exp_params["TRANSLATION_NOISE"],
        )

        simple_environment.next_object()

    print("\n\n===BEGGINING INFERENCE===")
    # Set LM mode appropriately, and re-set environment properties
    hdcd_lm.set_mode("inference")

    simple_environment.reset_object_iter()
    simple_environment.reset_sensations()

    # ***Iterate through the objects for INFERENCE***
    results_dic = dict(
        pose_recovered=0,
        correctly_converged=0,
        wrong_converged=0,
        no_match=0,
        time_for_inference=None,
        time_for_sensations=0,
        time_for_pose_estimate=0,
    )

    time_tracker = time.time()

    for __ in range(num_objects):
        print(
            f"\n\nCurrently INFERRING on object: "
            f"{simple_environment.current_object.object_id}"
        )

        hdcd_lm.reset_lm_predictions()

        # Feed the learning module the next sensation of the object
        # Note we may only sense a subset of the learned features at inference,
        # but that in this implementation, we will always explore e.g. the first
        # 10 features, rather than a random subset (not esp. significant
        # for synthetic objects)
        pre_sensation_time = time.time()
        for __ in range(exp_params["NUM_F_FOR_SENSING"]):

            # NB current sensation will automatically finish by ensuring that
            # on the next sensation, we will point to the next object
            hdcd_lm.process_sensation(
                simple_environment.current_sensation(),
                alignment_threshold=exp_params["SDR_SENSATION_THRESHOLD"],
            )

        sensation_time = time.time() - pre_sensation_time

        print("\nTime used on sensations and weight operations: " + str(sensation_time))
        results_dic["time_for_sensations"] += sensation_time

        pre_pose_check_time = time.time()

        inferred_pose, inferred_sdr = hdcd_lm.get_dense_pose_and_sdr(
            clustering_method=exp_params["CLUSTERING_METHOD"],
            exp_params=exp_params,
        )

        if inferred_pose is not None:
            pose_recovered, _ = utils.check_pose_estimate(
                simple_environment.current_object.get_object_pose(),
                inferred_pose,
                euler_eta=exp_params["EULER_MATCH_ETA"],
                translation_eta=exp_params["TRANSLATION_MATCH_ETA"],
            )
        else:
            pose_recovered = False

        if pose_recovered:
            print("Successfully recovered the pose!")
        else:
            print("Failed to recover the pose...")

        post_pose_time = time.time() - pre_pose_check_time
        print(
            "\nTime for consensus determination and checking pose: "
            + str(post_pose_time)
        )
        results_dic["time_for_pose_estimate"] += post_pose_time

        pre_class_time = time.time()

        if inferred_sdr is not None:
            results_dic = utils.check_classification(
                learned_objects_dic,
                results_dic,
                target_id=simple_environment.current_object.object_id,
                inferred_sdr=inferred_sdr,
                sdr_classifier_threshold=exp_params["SDR_CLASSIFIER_THRESHOLD"],
            )
        else:
            results_dic["no_match"] += 1

        post_class_time = time.time() - pre_class_time
        print("\nTime for checking classification: " + str(post_class_time))

        results_dic["pose_recovered"] += int(pose_recovered)

        # Get the next object; NB this automatically resets the sensation iter
        simple_environment.next_object()

        total_inner_time = sensation_time + post_class_time + post_pose_time
        print(
            "\nTotal time for main operations of object inference: "
            + str(total_inner_time)
        )

        print(
            "Proportion of time for sensations and weight operations: "
            + str(round(100 * sensation_time / total_inner_time, 2))
        )
        print(
            "Proportion of time for consensus determination: "
            + str(round(100 * post_pose_time / total_inner_time, 2))
        )

    results_dic["time_for_inference"] = time.time() - time_tracker

    print("\nTotal time for inference: " + str(results_dic["time_for_inference"]))

    print("\n\n\n===EVALUATION RESULTS===")
    print("Classification results:")
    print(
        "Classification accuracy: "
        + str(results_dic["correctly_converged"] / num_objects)
    )
    print(
        "Wrong convergence rate: " + str(results_dic["wrong_converged"] / num_objects)
    )
    print("No matches rate: " + str(results_dic["no_match"] / num_objects))
    print(
        "\nPose recovery accuracy: " + str(results_dic["pose_recovered"] / num_objects)
    )

    return results_dic
