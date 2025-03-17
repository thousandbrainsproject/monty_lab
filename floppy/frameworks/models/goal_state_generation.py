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
from tbp.monty.frameworks.models.goal_state_generation import EvidenceGoalStateGenerator


class FlopCountingEvidenceGoalStateGenerator(EvidenceGoalStateGenerator):
    """Extension of EvidenceGoalStateGenerator that counts FLOPs for KDTree operations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flop_counter = None

    def _compute_graph_mismatch(self):
        """Override to count FLOPs during graph mismatch computation.

        Similar to parent method but adds FLOP counting for KDTree operations.
        """
        logging.debug("Proposing an evaluation location based on graph mismatch")

        top_id, second_id = self.parent_lm.get_top_two_mlh_ids()

        top_mlh = self.parent_lm.get_mlh_for_object(top_id)
        second_mlh_object = self.parent_lm.get_mlh_for_object(second_id)

        top_mlh_graph = self.parent_lm.get_graph(top_id, input_channel="first").pos

        if self.focus_on_pose:
            second_id = top_id
            _, second_mlh = self.parent_lm.get_top_two_pose_hypotheses_for_graph_id(
                top_id
            )
        else:
            second_mlh = second_mlh_object

        # Transform graph coordinates as in parent method
        rotated_graph = top_mlh["rotation"].inv().apply(top_mlh_graph)
        current_mlh_location = top_mlh["rotation"].inv().apply(top_mlh["location"])
        top_mlh_graph = rotated_graph - current_mlh_location
        top_mlh_graph = (
            second_mlh["rotation"].apply(top_mlh_graph) + second_mlh["location"]
        )
        # Perform KDTree search
        radius_node_dists = self.parent_lm.get_graph(
            second_id, input_channel="first"
        ).find_nearest_neighbors(
            top_mlh_graph,
            num_neighbors=1,
            return_distance=True,
        )

        # Calculate FLOPs for KDTree operations
        num_search_points = len(top_mlh_graph)
        num_reference_points = len(
            self.parent_lm.get_graph(second_id, input_channel="first").pos
        )

        # Tree Traversal FLOPs
        dim1 = top_mlh_graph.shape[1]
        traversal_flops = num_search_points * dim1 * np.log2(num_reference_points)
        # FLOPs for distance
        num_examined_points = int(np.log2(num_reference_points))
        distance_flops = (
            num_search_points * num_examined_points * (3 * dim1 + dim1 + 1)
        )  # dim1*(3 ops per dim) + dim1 additions + 1 sqrt
        # Heap operations
        heap_flops = 0  # because num_neighbors is 1
        # FLOPs for bounding box check
        bounding_box_flops = num_search_points * num_examined_points * dim1
        total_flops = int(
            traversal_flops + distance_flops + heap_flops + bounding_box_flops
        )

        # Add FLOPs to counter if available
        if self.flop_counter is not None:
            self.flop_counter.add_flops(total_flops)

        target_loc_id = np.argmax(radius_node_dists)
        target_loc_separation = np.max(radius_node_dists)

        self.prev_top_mlhs = [top_mlh, second_mlh_object]

        return target_loc_id, target_loc_separation
