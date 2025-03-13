# Copyright 2025 Thousand Brains Project
# Copyright 2023 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""
Figure 4: Visualize 8-patch view finder
"""

import copy
import os
from numbers import Number
from pathlib import Path
from typing import (
    Container,
    List,
    Optional,
    Union,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
from data_utils import (
    DMC_ANALYSIS_DIR,
    DMC_RESULTS_DIR,
    VISUALIZATION_RESULTS_DIR,
    DetailedJSONStatsInterface,
    get_frequency,
    load_eval_stats,
)
from matplotlib.lines import Line2D
from plot_utils import TBP_COLORS, add_legend, axes3d_set_aspect_equal, violinplot

plt.rcParams["font.size"] = 8
plt.rcParams["font.family"] = "Arial"
plt.rcParams["svg.fonttype"] = "none"

OUT_DIR = DMC_ANALYSIS_DIR / "fig6"
OUT_DIR.mkdir(parents=True, exist_ok=True)
