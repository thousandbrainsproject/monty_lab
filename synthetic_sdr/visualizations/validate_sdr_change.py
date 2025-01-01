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
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import seaborn as sns
import torch

sys.path.append("./")
from utils import set_seed  # noqa: E402

sns.set(style="dark")
set_seed(0)


def plot_gif_stripplot(vals, vmin=0, vmax=2048):
    """
    Creates a stripplot to visualize the SDR change over time
    """

    fig, ax = plt.subplots(figsize=(5, 10))

    def animate(i):
        ax.clear()
        sns.stripplot(vals[i], size=5, jitter=False, ax=ax)
        ax.set_ylim([vmin, vmax])
        ax.set_xlabel("Object SDRs")
        ax.set_ylabel("SDR Bits")
        ax.text(
            0.5,
            1.05,
            "{0}/{1}".format(i + 1, len(vals)),
            ha="center",
            va="center",
            fontsize=20,
            transform=ax.transAxes,
        )
        return ax

    return animation.FuncAnimation(fig, animate, frames=len(vals))


# obj sdrs
exp_path = "logs/exp_001/seed_000/pth"
preds = torch.load(os.path.join(exp_path, "preds.pth"))

vals_list = []
obj_sdrs_change = preds["sparse"]
for obj in obj_sdrs_change[::10]:
    vals = torch.where(obj == 1.0)[1][:410]
    vals = torch.chunk(vals, 10)
    vals = [val.numpy().tolist() for val in vals]
    vals_list.append(vals)

ani = plot_gif_stripplot(vals_list)
ani.save(os.path.join(exp_path, "SDR_change.gif"), writer="pillow", fps=10)
plt.close()
