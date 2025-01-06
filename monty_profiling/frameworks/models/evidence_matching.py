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
        self.reset_flop_counters()

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
