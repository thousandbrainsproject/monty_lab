# Copyright 2025 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import argparse

from tbp.monty.frameworks.run import run

from experiments import CONFIGS


def run_temporal_memory(all_configs, experiments=None):
    """Temporal memory runner that calls the standard run().

    # TODO: standardize monty configs and method of running
    """
    if not experiments:
        cmd_parser = argparse.ArgumentParser()
        cmd_parser.add_argument(
            "-e",
            "--experiments",
            choices=all_configs,
            nargs="+",
            help="Experiment names",
        )
        cmd_args = cmd_parser.parse_args()

        experiments = cmd_args.experiments

    for exp_name in experiments:
        run(all_configs[exp_name])


if __name__ == "__main__":
    run_temporal_memory(all_configs=CONFIGS)
if __name__ == "__main__":
    run_temporal_memory(all_configs=CONFIGS)
if __name__ == "__main__":
    run_temporal_memory(all_configs=CONFIGS)
