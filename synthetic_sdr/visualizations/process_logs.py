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
import json
import os
import sys
from glob import glob

import matplotlib.pyplot as plt
import torch
import yaml
from tqdm import tqdm

sys.path.append("./")
from utils import get_gif, get_gif_linedot, get_overlaps  # noqa: E402

# parse arguments
parser = argparse.ArgumentParser(description="Script to create some visualizations")
parser.add_argument(
    "--path",
    "-p",
    type=str,
    help="The path of the experiment logs (i.e., pth directory)",
)
args = parser.parse_args()
experiments_paths = glob(args.path)

for path in tqdm(experiments_paths):

    # extract some information for faster processing
    configs = yaml.safe_load(open(os.path.join(path, "configs.yaml"), "r"))
    scalars_path = os.path.join(path, "scalars.pth")
    raw_overlaps = torch.load(scalars_path)["overlap_distance"]

    overlap = torch.stack(raw_overlaps[-10:]).mean()
    json.dump(
        {"overlap_distance": overlap.item()},
        open(os.path.join(path, "overlap.json"), "w"),
    )

    overlap_full = torch.stack(raw_overlaps)
    json.dump(
        {"overlap_distance": overlap_full.numpy().tolist()},
        open(os.path.join(path, "overlap_full.json"), "w"),
    )

    # === GIFS === #
    pred_path = os.path.join(path, "preds.pth")
    pred_file = torch.load(pred_path)
    fps_preds = 10

    # sparse gif
    if len(pred_file["sparse"]):
        sparse_sims = [
            get_overlaps(pred_file["sparse"][i])
            for i in range(0, len(pred_file["sparse"]), 10)
        ]
        anim = get_gif(sparse_sims, *configs["overlap_range"])
        anim.save(
            os.path.join(path, "pred_sparse.gif"), writer="imagemagick", fps=fps_preds
        )

    # targets gif
    target_path = os.path.join(path, "target.pth")
    target_file = torch.load(target_path)

    if len(target_file["target_overlaps"]):
        vals = target_file["target_overlaps"]
        anim = get_gif(vals, *configs["overlap_range"])
        anim.save(
            os.path.join(path, "target_overlaps.gif"),
            writer="imagemagick",
            fps=len(vals) * fps_preds / len(sparse_sims),
        )

    # linedot gif
    overlap = torch.stack(raw_overlaps[::100])
    ani = get_gif_linedot(overlap)
    ani.save(os.path.join(path, "linedot.gif"), writer="pillow", fps=fps_preds)

    plt.close()
