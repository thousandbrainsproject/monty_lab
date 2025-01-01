# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import argparse
import os

import torch
import yaml
from datasets import ObjectsDataset
from tqdm import tqdm

from loggers import Accumulator, PTHLogger
from models import Model
from utils import checkdir, get_dists, map_dists, set_seed

# Manually set arguments
base_object_dimensionality = 16

# parse arguments
parser = argparse.ArgumentParser(description="Main script")
parser.add_argument("--config", "-i", type=str, help="The config file path")
parser.add_argument(
    "--logs", "-o", default="./logs", type=str, help="The config file path"
)
parser.add_argument("--seed", "-s", default=0, type=int, help="The experiment seed")
args = parser.parse_args()

# load experiment configuration
try:
    configs = yaml.safe_load(open(args.config, "r"))
except Exception():
    raise Exception("Error opening config file")

log_path = os.path.join(
    args.logs,
    os.path.splitext(os.path.basename(args.config))[0],
    f"seed_{str(args.seed).zfill(3)}",
)


# create synthetic data
set_seed(args.seed)
synthetic_objs = torch.randn(0, base_object_dimensionality)

# initialize dataset and model classes
dataset = ObjectsDataset(batch_size=configs["batch_size"])
model = Model(
    sdr_length=configs["sdr_length"],
    sdr_on_bits=configs["sdr_on_bits"],
    lr=configs["lr"],
)

# logging initialization
checkdir(log_path)
accumulator = Accumulator(n_vals=2)
pth_writer = PTHLogger(log_path, configs, do_log=configs["log_pth"])


def add_objects(synthetic_objs, new_objects):
    """
    adds new objects to the dataset class and the model

    * synthetic_objs keep track of the object representations that are used to
    create the similarity matrix and target overlap.
    This is only used to create the synthetic similarity.
    Each synthetic object is a dense vector of dimension `base_object_dimensionality`.


    * dataset.add_objects: informs the dataset that it should now sample
    from more indices because synthetic_objs now has more entries.

    * model.add_objects: informs the model to create more embeddings and
    updates the optimizer parameters because it will receive larger indices
    from the dataset now and it needs to represent those with new SDRs.
    """

    synthetic_objs = torch.cat(
        [synthetic_objs, torch.randn(new_objects, base_object_dimensionality)]
    )
    dataset.add_objects(new_objects)
    model.add_objects(new_objects)
    return synthetic_objs


# Loop for adding objects in a streaming manner
for new_objects in configs["num_objects"]:

    # add objects and recalculate target overlaps
    synthetic_objs = add_objects(synthetic_objs, new_objects)
    target_overlaps = map_dists(
        get_dists(synthetic_objs).clone(),
        *configs["evidence_range"],
        *configs["overlap_range"],
    )

    pth_writer.log_targets(
        {
            "target_objs": synthetic_objs.detach().cpu(),
            "target_overlaps": target_overlaps.detach().cpu(),
        }
    )

    # main training loop over num_epochs
    for e in tqdm(range(configs["num_epochs"])):

        # training of batches from the dataset
        for ix in dataset:

            results = model.train_reps(
                ix=ix,
                target_overlaps=target_overlaps[ix, :][:, ix],
            )
            accumulator.append(results)

        # average results over all batches in one epoch before
        # sending them to the logging functions
        avg_results = accumulator.get_avg(results)
        pth_writer.log_results(avg_results)
        accumulator.reset()

        # epoch level logging
        if e % configs["log_every"] == 0:
            pth_writer.log_figs(
                {"sparse": model.infer_rep(model.obj_reps.weight.data.clone())},
            )


# close writers/loggers
pth_writer.close()
print(f"finished experiment {args.config} with seed {args.seed}")
