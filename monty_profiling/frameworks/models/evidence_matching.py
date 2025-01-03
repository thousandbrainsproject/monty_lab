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
    -   _get_input_channels_in_graph
    -   _get_initial_hypothesis_space
    -   _add_hypotheses_to_hpspace
    -   _get_evidence_update_threshold
    -   _calculate_evidence_for_new_locations

    "Normal" np methods to catch in FlopCounter:
    -   np.clip
    - np.max and np.min
    - np.where
    - np.round
    - np.dot
    """
