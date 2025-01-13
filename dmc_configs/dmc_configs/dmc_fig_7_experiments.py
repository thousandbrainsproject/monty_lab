"""
Implements `dist_agent_1lm_randrot_nohyp_x_percent_20p`
"""

import copy
import os
from pathlib import Path

import numpy as np
from .dmc_eval_experiments import dist_agent_1lm_nohyp_randrot

dist_agent_1lm_nohyp_randrot_x_percent_20p = copy.deepcopy(dist_agent_1lm_nohyp_randrot)
