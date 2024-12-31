# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os

import torch
import yaml


class Accumulator:
    """
    Custom class for accumulating values over time and averaging them.
    """

    def __init__(self, operation="avg", n_vals=5):
        self.operation = operation
        self.n_vals = n_vals
        self.reset()

    def append(self, results):
        assert (
            len(results) == self.n_vals
        ), "accumulator received different number of results"

        for i, (_, v) in enumerate(results.items()):
            self.vals[i].append(v)

    def get_avg(self, results):
        for i, (k, _) in enumerate(results.items()):
            if torch.is_tensor(self.vals[i][0]):
                results[k] = torch.stack(self.vals[i]).mean()
            else:
                results[k] = torch.mean(torch.tensor(self.vals[i]))

        return results

    def reset(self):
        self.vals = [[] for _ in range(self.n_vals)]


class PTHLogger:
    """
    Custom class for logging training results over time.
    Logs will be used later for visualizations
    """

    def __init__(self, path, configs, do_log=True):
        self.path = os.path.join(path, "pth")
        os.makedirs(self.path)
        self.scalars = {}
        self.preds = {"dense": [], "sparse": []}
        self.targets = {"target_objs": [], "target_overlaps": []}

        self.configs = configs
        self.do_log = do_log
        self.log_configs()

    def log_configs(self):
        if not self.do_log:
            return

        yaml.dump(self.configs, open(os.path.join(self.path, "configs.yaml"), "w"))

    def log_targets(self, results):
        if not self.do_log:
            return

        if "target_objs" in results:
            self.targets["target_objs"].append(results["target_objs"])
        if "target_overlaps" in results:
            self.targets["target_overlaps"].append(results["target_overlaps"])

    def log_results(self, results):
        if not self.do_log:
            return

        for k, v in results.items():
            if k not in self.scalars:
                self.scalars[k] = []
            self.scalars[k].append(v)

    def log_figs(self, results):
        if not self.do_log:
            return

        if "dense" in results:
            self.preds["dense"].append(results["dense"])
        if "sparse" in results:
            self.preds["sparse"].append(results["sparse"])

    def close(self):
        if not self.do_log:
            return

        torch.save(self.scalars, os.path.join(self.path, "scalars.pth"))
        torch.save(self.preds, os.path.join(self.path, "preds.pth"))
        torch.save(self.targets, os.path.join(self.path, "target.pth"))
