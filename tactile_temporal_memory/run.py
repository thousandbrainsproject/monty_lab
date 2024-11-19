# Copyright 2023 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# Load all experiment configurations from local project
from config import CONFIGS
from tbp.monty.frameworks.run import main

if __name__ == "__main__":
    main(all_configs=CONFIGS)
    # main(all_configs=CONFIGS, experiments=["explore_touch_test"])
    # main(all_configs=CONFIGS, experiments=["tactile_TM_test"])
