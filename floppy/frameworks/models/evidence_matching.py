# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import logging

import numpy as np
from scipy.spatial import KDTree
from tbp.monty.frameworks.models.evidence_matching import (
    EvidenceGraphLM,
)
from tbp.monty.frameworks.utils.graph_matching_utils import (
    get_custom_distances,
    get_relevant_curvature,
)
from tbp.monty.frameworks.utils.spatial_arithmetics import (
    rotate_pose_dependent_features,
)

from frameworks.models.goal_state_generation import (
    FlopCountingEvidenceGoalStateGenerator,
)


class FlopCountingEvidenceGraphLM(EvidenceGraphLM):
    """Extension of EvidenceGraphLM that counts FLOPs for KDTree operations."""

    def __init__(
        self,
        max_match_distance,
        tolerances,
        feature_weights,
        feature_evidence_increment=1,
        max_nneighbors=3,
        initial_possible_poses="informed",
        evidence_update_threshold="all",
        vote_evidence_threshold=0.8,
        past_weight=1,
        present_weight=1,
        vote_weight=1,
        object_evidence_threshold=1,
        x_percent_threshold=10,
        path_similarity_threshold=0.1,
        pose_similarity_threshold=0.35,
        required_symmetry_evidence=5,
        graph_delta_thresholds=None,
        max_graph_size=0.3,
        max_nodes_per_graph=2000,
        num_model_voxels_per_dim=50,
        use_multithreading=True,
        gsg_class=FlopCountingEvidenceGoalStateGenerator,
        gsg_args=None,
        *args,
        **kwargs,
    ):
        # Call parent's __init__ with all parameters
        super().__init__(
            max_match_distance=max_match_distance,
            tolerances=tolerances,
            feature_weights=feature_weights,
            feature_evidence_increment=feature_evidence_increment,
            max_nneighbors=max_nneighbors,
            initial_possible_poses=initial_possible_poses,
            evidence_update_threshold=evidence_update_threshold,
            vote_evidence_threshold=vote_evidence_threshold,
            past_weight=past_weight,
            present_weight=present_weight,
            vote_weight=vote_weight,
            object_evidence_threshold=object_evidence_threshold,
            x_percent_threshold=x_percent_threshold,
            path_similarity_threshold=path_similarity_threshold,
            pose_similarity_threshold=pose_similarity_threshold,
            required_symmetry_evidence=required_symmetry_evidence,
            graph_delta_thresholds=graph_delta_thresholds,
            max_graph_size=max_graph_size,
            max_nodes_per_graph=max_nodes_per_graph,
            num_model_voxels_per_dim=num_model_voxels_per_dim,
            use_multithreading=use_multithreading,
            gsg_class=gsg_class,
            gsg_args=gsg_args,
            *args,
            **kwargs,
        )
        self.use_multithreading = False
        self.flop_counter = None

    def _update_evidence_with_vote(self, state_votes, graph_id):
        """Count FLOPs for KDTree operations during vote processing."""
        # Extract vote information
        graph_location_vote = np.zeros((len(state_votes), 3))
        vote_evidences = np.zeros(len(state_votes))
        for n, vote in enumerate(state_votes):
            graph_location_vote[n] = vote.location
            vote_evidences[n] = vote.confidence

        # Execute KDTree query
        vote_location_tree = KDTree(
            graph_location_vote,
            leafsize=40,
        )
        vote_nn = 3  # TODO: Make this a parameter?
        vote_nn = min(graph_location_vote.shape[0], vote_nn)

        # Account for FLOPs for KDTree Construction
        num_reference_points = len(graph_location_vote)
        dim1 = self.possible_locations[graph_id].shape[1]
        kdtree_flops = int(num_reference_points * np.log2(num_reference_points) * dim1)

        (radius_node_dists, radius_node_ids) = vote_location_tree.query(
            self.possible_locations[graph_id],
            k=vote_nn,
            p=2,
            workers=1,
        )

        # Account for FLOPs for KDTree query
        num_search_points = len(self.possible_locations[graph_id])
        num_reference_points = len(graph_location_vote)

        # Tree Traversal FLOPs
        # Depth of tree is log2(num_reference_points) for balanced tree
        dim1 = self.possible_locations[graph_id].shape[1]
        traversal_flops = num_search_points * dim1 * np.log2(num_reference_points)
        # FLOPs for distance
        num_examined_points = int(np.log2(num_reference_points))
        distance_flops = (
            num_search_points * num_examined_points * (3 * dim1 + dim1 + 1)
        )  # dim1*(3 ops per dim) + dim1 additions + 1 sqrt
        # Heap operations
        heap_flops = num_search_points * num_examined_points * np.log2(max(vote_nn, 1))

        # FLOPs for bounding box check
        bounding_box_flops = num_search_points * num_examined_points * dim1
        total_flops = int(
            kdtree_flops
            + traversal_flops
            + distance_flops
            + heap_flops
            + bounding_box_flops
        )
        if self.flop_counter is not None:
            self.flop_counter.add_flops(total_flops)

        # Remainder of code
        if vote_nn == 1:
            radius_node_dists = np.expand_dims(radius_node_dists, axis=1)
            radius_node_ids = np.expand_dims(radius_node_ids, axis=1)

        radius_evidences = vote_evidences[radius_node_ids]
        node_distance_weights = self._get_node_distance_weights(radius_node_dists)
        too_far_away = node_distance_weights <= 0
        all_radius_evidence = np.ma.array(radius_evidences, mask=too_far_away)
        distance_weighted_vote_evidence = np.ma.max(all_radius_evidence, axis=1)

        # Update evidence based on weights
        if self.past_weight + self.present_weight == 1:
            self.evidence[graph_id] = np.ma.average(
                [self.evidence[graph_id], distance_weighted_vote_evidence],
                weights=[1, self.vote_weight],
                axis=0,
            )
        else:
            self.evidence[graph_id] = np.ma.sum(
                [
                    self.evidence[graph_id],
                    distance_weighted_vote_evidence * self.vote_weight,
                ],
                axis=0,
            )

    def _calculate_evidence_for_new_locations(
        self,
        graph_id,
        input_channel,
        search_locations,
        features,
        hyp_ids_to_test,
    ):
        """Use search locations, sensed features and graph model to calculate evidence.

        First, the search locations are used to find the nearest nodes in the graph
        model. Then we calculate the error between the stored pose features and the
        sensed ones. Additionally we look at whether the non-pose features match at the
        neigboring nodes. Everything is weighted by the nodes distance from the search
        location.
        If there are no nodes in the search radius (max_match_distance), evidence = -1.

        We do this for every incoming input channel and its features if they are stored
        in the graph and take the average over the evidence from all input channels.

        Returns:
            The location evidence.
        """
        logging.debug(
            f"Calculating evidence for {graph_id} using input from " f"{input_channel}"
        )

        pose_transformed_features = rotate_pose_dependent_features(
            features[input_channel],
            self.possible_poses[graph_id][hyp_ids_to_test],
        )

        # Get max_nneighbors nearest nodes to search locations.
        nearest_node_ids = self.get_graph(
            graph_id, input_channel
        ).find_nearest_neighbors(
            search_locations,
            num_neighbors=self.max_nneighbors,
        )

        # def find_nearest_neighbors():
        # (distances, nearest_node_ids) = self._location_tree.query(
        #     search_locations,
        #     k=num_neighbors,
        #     p=2,  # eucledian distance
        #     workers=1,  # using more than 1 worker slows down run on lambda.
        # )

        num_search_points = len(search_locations)
        num_reference_points = len(
            self.graph_memory.get_locations_in_graph(graph_id, input_channel)
        )
        dim1 = search_locations.shape[1]
        traversal_flops = num_search_points * dim1 * np.log2(num_reference_points)
        num_examined_points = int(np.log2(num_reference_points))
        distance_flops = (
            num_search_points * num_examined_points * (3 * dim1 + dim1 + 1)
        )  # dim1*(3 ops per dim) + dim1 additions + 1 sqrt
        heap_flops = (
            num_search_points * num_examined_points * np.log2(self.max_nneighbors)
        )
        bounding_box_flops = num_search_points * num_examined_points * dim1
        total_flops = int(
            traversal_flops + distance_flops + heap_flops + bounding_box_flops
        )
        if self.flop_counter is not None:
            self.flop_counter.add_flops(total_flops)

        if self.max_nneighbors == 1:
            nearest_node_ids = np.expand_dims(nearest_node_ids, axis=1)

        nearest_node_locs = self.graph_memory.get_locations_in_graph(
            graph_id, input_channel
        )[nearest_node_ids]
        max_abs_curvature = get_relevant_curvature(features[input_channel])
        custom_nearest_node_dists = get_custom_distances(
            nearest_node_locs,
            search_locations,
            pose_transformed_features["pose_vectors"][:, 0],
            max_abs_curvature,
        )

        # shape=(H, K)
        node_distance_weights = self._get_node_distance_weights(
            custom_nearest_node_dists
        )
        # Get IDs where custom_nearest_node_dists > max_match_distance
        mask = node_distance_weights <= 0

        new_pos_features = self.graph_memory.get_features_at_node(
            graph_id,
            input_channel,
            nearest_node_ids,
            feature_keys=["pose_vectors", "pose_fully_defined"],
        )
        # Calculate the pose error for each hypothesis
        # shape=(H, K)
        radius_evidence = self._get_pose_evidence_matrix(
            pose_transformed_features,
            new_pos_features,
            input_channel,
            node_distance_weights,
        )
        # Set the evidences which are too far away to -1
        radius_evidence[mask] = -1
        # If a node is too far away, weight the negative evidence fully (*1). This
        # only comes into play if there are no nearby nodes in the radius, then we
        # want an evidence of -1 for this hypothesis.
        # NOTE: Currently we don't weight the evidence by distance so this doesn't
        # matter.
        node_distance_weights[mask] = 1

        # If no feature weights are provided besides the ones for point_normal
        # and curvature_directions we don't need to calculate feature evidence.
        if self.use_features_for_matching[input_channel]:
            # add evidence if features match
            node_feature_evidence = self._calculate_feature_evidence_for_all_nodes(
                features, input_channel, graph_id
            )
            hypothesis_radius_feature_evidence = node_feature_evidence[nearest_node_ids]
            # Set feature evidence of nearest neighbors that are too far away to 0
            hypothesis_radius_feature_evidence[mask] = 0
            # Take the maximum feature evidence out of the nearest neighbors in the
            # search radius and weighted by its distance to the search location.
            # Evidence will be in [0, 1] and is only 1 if all features match
            # perfectly and the node is at the search location.
            radius_evidence = (
                radius_evidence
                + hypothesis_radius_feature_evidence * self.feature_evidence_increment
            )
        # We take the maximum to be better able to deal with parts of the model where
        # features change quickly and we may have noisy location information. This way
        # we check if we can find a good match of pose features within the search
        # radius. It doesn't matter if there are also points stored nearby in the model
        # that are not a good match.
        # Removing the comment weights the evidence by the nodes distance from the
        # search location. However, epirically this did not seem to help.
        # shape=(H,)
        location_evidence = np.max(
            radius_evidence,  # * node_distance_weights,
            axis=1,
        )
        return location_evidence
