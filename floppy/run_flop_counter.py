# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from tbp.monty.frameworks.run_env import setup_env

setup_env()

import sys
from pathlib import Path

monty_lab_path = Path("~/tbp/monty_lab").expanduser().resolve()
sys.path.append(str(monty_lab_path))
from dmc.configs import CONFIGS  # noqa: E402

from frameworks.run import flop_main  # noqa: E402

if __name__ == "__main__":
    flop_main(all_configs=CONFIGS)
