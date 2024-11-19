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
from experiments.supervised_config_args import (
    DataClassArgumentParser,
    merge_dataclass_args,
)


def run_supervised(all_configs, experiments=None):
    """A temporary alternative to main with the same arguments used for supervised exps.

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

        exp = all_configs[exp_name]
        exp_class = exp["experiment_class"]
        exp_parser = DataClassArgumentParser(exp_class.DEFAULT_ARGS)
        exp_args = merge_dataclass_args(exp_parser.parse_dict(exp))

        # conform to existing run function
        exp_dict = dict(experiment_class=exp_class)
        exp_dict.update(exp_args.__dict__)
        run(exp_dict)


if __name__ == "__main__":
    run_supervised(all_configs=CONFIGS)
