from scipy.spatial import KDTree
import numpy as np
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
from tbp.monty.frameworks.models.evidence_matching import EvidenceGraphLM


class FlopCountingEvidenceGraphLM(EvidenceGraphLM):
    """Extension of EvidenceGraphLM that counts FLOPs for computationally expensive operations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    """
    Notes on _update_evidence:
    - No "special" function directly called to compute FLOPs.
    - Calls the following subroutines (which may need to be computed for FLOPs):
    - [x]  _get_input_channels_in_graph = not relevant for FLOPs
    -  _get_initial_hypothesis_space = critical
    -   _add_hypotheses_to_hpspace
    -   _get_evidence_update_threshold
    -   _calculate_evidence_for_new_locations
    """

    """
    Note on _get_initial_hypothesis_space:
    - Doesn't have special numpy methods to catch in FlopCounter
    - Calls the following subroutines (which may need to be computed for FLOPs):
    - [x] self._get_all_informed_possible_poses(graph_id, features, input_channel)
    - self._calculate_feature_evidence_for_all_nodes

    Notes on _get_all_informed_possible_poses:
    - [x] get_more_directions_in_plane
    - [x] align_multiple_orthonormal_vectors
        - Ignoring Rotation.from_matrix, because it likely deals with small 3x3 matrices

    Notes on spatial_arithmetics.py:
    - rotations_to_quats
    - rot_mats_to_quats
    - euler_to_quats
    The above three are ignored despite `.inv()` because they are operating on small 3x3 matrices
    """

    def _update_evidence_with_vote(self, state_votes, graph_id):
        """Use incoming votes to update all hypotheses. Counts FLOPs for KDTree operations."""
        # Extract information from list of State classes into np.arrays for efficient
        # matrix operations and KDTree search.
        graph_location_vote = np.zeros((len(state_votes), 3))
        vote_evidences = np.zeros(len(state_votes))
        for n, vote in enumerate(state_votes):
            graph_location_vote[n] = vote.location
            vote_evidences[n] = vote.confidence

        vote_location_tree = KDTree(
            graph_location_vote,
            leafsize=40,
        )
        vote_nn = 3  # TODO: Make this a parameter?
        if graph_location_vote.shape[0] < vote_nn:
            vote_nn = graph_location_vote.shape[0]

        # Count FLOPs for KDTree query
        # For each query point:
        # - Calculate distance to each reference point: 5 FLOPs per coordinate (3 coords)
        # - Sort distances to find k nearest: O(n log k) comparisons
        num_query_points = len(self.possible_locations[graph_id])
        num_ref_points = len(graph_location_vote)

        distance_flops = (
            num_query_points * num_ref_points * 15
        )  # 5 FLOPs * 3 coordinates
        sorting_flops = num_query_points * num_ref_points * np.log2(vote_nn)
        total_kdtree_flops = distance_flops + sorting_flops

        # Add to FLOP counter
        if not hasattr(self, "kdtree_flops"):
            self.kdtree_flops = defaultdict(int)
        self.kdtree_flops["query"] += int(total_kdtree_flops)

        # Perform the actual query
        (radius_node_dists, radius_node_ids) = vote_location_tree.query(
            self.possible_locations[graph_id],
            k=vote_nn,
            p=2,
            workers=1,
        )

        # Rest of the function remains the same as original
        if vote_nn == 1:
            radius_node_dists = np.expand_dims(radius_node_dists, axis=1)
            radius_node_ids = np.expand_dims(radius_node_ids, axis=1)
        radius_evidences = vote_evidences[radius_node_ids]
        node_distance_weights = self._get_node_distance_weights(radius_node_dists)
        too_far_away = node_distance_weights <= 0
        all_radius_evidence = np.ma.array(radius_evidences, mask=too_far_away)
        distance_weighted_vote_evidence = np.ma.max(
            all_radius_evidence,
            axis=1,
        )

        if self.past_weight + self.present_weight == 1:
            self.evidence[graph_id] = np.ma.average(
                [
                    self.evidence[graph_id],
                    distance_weighted_vote_evidence,
                ],
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
