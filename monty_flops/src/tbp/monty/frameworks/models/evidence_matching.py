from tbp.monty.frameworks.models.evidence_matching import EvidenceGraphLM
from typing import Any, Dict
from playground.flop_counter import count_flops


class FlopCountingEvidenceGraphLM(EvidenceGraphLM):
    """Wrapper class that adds FLOP counting to EvidenceGraphLM"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_flops = 0

    @count_flops
    def _get_pose_evidence_matrix(self, *args, **kwargs):
        result = super()._get_pose_evidence_matrix(*args, **kwargs)
        self.total_flops += _get_pose_evidence_matrix.flops
        return result
