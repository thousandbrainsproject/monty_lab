# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os
from argparse import ArgumentParser
from dataclasses import dataclass, field, fields, is_dataclass, make_dataclass
from typing import Callable, Dict, Union

import torch
from modelnet import ModelNet40
from tbp.monty.frameworks.config_utils.make_dataset_configs import ExperimentArgs


@dataclass
class SupervisedExpArguments:

    model_class: Callable  # TODO: maybe make vector neurons the default
    model_args: Union[Dict, dataclass]  # and vector neuron args the default
    optimizer_class: Callable = field(default=torch.optim.Adam)
    optimizer_args: Union[Dict, dataclass] = field(
        default_factory=lambda: dict(lr=1e-3, weight_decay=1e-4)
    )
    schedulers: Dict = field(
        default_factory=lambda: dict(lr_scheduler=torch.optim.lr_scheduler.StepLR)
    )
    schedulers_args: Dict[str, Union[Dict, dataclass]] = field(
        default_factory=lambda: dict(lr_scheduler=dict(step_size=20, gamma=0.7))
    )
    loss_function: Callable = field(default=torch.nn.functional.nll_loss)
    experiment_args: Union[Dict, dataclass] = field(default=ExperimentArgs())
    dataset_class: Callable = field(default=ModelNet40)
    dataset_args: Dict = field(
        default_factory=lambda: dict(
            transform=None, root=os.path.expanduser("~/tbp/datasets/modelnet40")
        )
    )
    train_dataloader_args: Dict = field(
        default_factory=lambda: dict(
            batch_size=16,
            shuffle=True
        )
    )
    eval_dataloader_args: Dict = field(
        default_factory=lambda: dict(
            batch_size=16,
            shuffle=True
        )
    )
    eval_scenarios: Dict = field(
        default_factory=lambda: dict(
            default_scenario=None
        )
    )


@dataclass
class OnlineOptimizationExpArguments:

    eval_metrics: Dict
    model_class: Callable  # TODO: set default to be icp
    model_args: Dict
    dataset_class: Callable
    dataset_args: Dict
    dataloader_args: Dict = field(
        default_factory=lambda: dict(batch_size=1, shuffle=True)
    )
    experiment_args: Union[Dict, dataclass] = field(default=ExperimentArgs())
    eval_scenarios: Dict = field(
        default_factory=lambda: dict(
            default_scenario=None
        )
    )
    compose: bool = False


class DataClassArgumentParser(ArgumentParser):
    """Use type hints on dataclasses to generate arguments.

    # NOTE copied from projects/graph_classificaion/parser_utils

    The class is designed to play well with the native argparse. In particular, you can
    add more (non-dataclass backed) arguments to the parser after initialization and
    you'll get the output back after parsing as an additional namespace.

    Adapted from HuggingFace HFArgumentParser. Removed type checking.
    """

    def __init__(self, dataclass_types, **kwargs):
        """Initialize the parser.

        Args:
            dataclass_types:
                Dataclass type, or list of dataclass types for which we will "fill"
                instances with the parsed args.
            kwargs:
                (Optional) Passed to `argparse.ArgumentParser()` in the regular way.
        """
        super().__init__(**kwargs)
        if is_dataclass(dataclass_types):
            dataclass_types = [dataclass_types]
        self.dataclass_types = dataclass_types
        for dtype in self.dataclass_types:
            self._add_dataclass_arguments(dtype)

    def _add_dataclass_arguments(self, dtype):
        for fld in fields(dtype):
            if not fld.init:
                continue
            fld_name = f"--{fld.name}"
            kwargs = fld.metadata.copy()
            self.add_argument(fld_name, **kwargs)

    def parse_dict(self, args):
        """Parses a dict and populating the dataclass types.

        Returns:
            A tuple of dataclass instances.
        """
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in fields(dtype) if f.init}
            inputs = {k: v for k, v in args.items() if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)
        return (*outputs,)


def merge_dataclass_args(dataclasses, class_name="Args"):
    """Merge multiple dataclasses into one.

    Returns:
        A merged dataclass instance.
    """
    fields, args = {}, {}
    for dc in dataclasses:
        fields.update(dc.__dataclass_fields__)
        args.update(dc.__dict__)

    return make_dataclass(cls_name="Args", fields=fields)(**args)
