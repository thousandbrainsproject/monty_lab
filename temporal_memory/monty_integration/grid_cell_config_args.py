# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# Used to be in frameworks/config_utils/

from tbp.monty.frameworks.models.htm_learning_modules import L4_and_L6a_3d_LM
from tbp.monty.frameworks.models.htm_monty_model import SingleLMMontyHTM

l4_and_l6a_3d_lm_base_config = dict(
    tm_num_minicolumns=1024,
    tm_num_cells_per_minicolumn=5,
    tm_proximal_w=11,
    tm_initial_permanence=0.51,
    tm_connected_permanence=0.5,
    tm_permanence_increment=0.1,
    tm_permanence_decrement=0.02,
    tm_seed=42,
    gc_num_modules_per_axis=10,
    gc_num_cells_per_axis_per_module=5,
    gc_cell_coordinate_offsets=(0.001, 0.999),
    gc_activation_threshold=8,
    gc_initial_permanence=0.51,
    gc_connected_permanence=0.5,
    gc_matching_threshold=8,
    gc_sample_size=-1,
    gc_permanence_increment=0.1,
    gc_permanence_decrement=0.02,
    gc_anchoring_method="corners",
    gc_random_location=False,
    gc_seed=42,
)

GridCellArgs = dict(
    monty_class=SingleLMMontyHTM,
    learning_module_configs=dict(
        learning_module_0=dict(
            learning_module_class=L4_and_L6a_3d_LM,
            learning_module_args=l4_and_l6a_3d_lm_base_config,
        )
    )
)
