# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os
import shutil

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F


def checkdir(path):
    """Removes existing log files before writing new ones"""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def set_seed(seed):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_sims(reps):
    """Returns cosine similarity of representations"""
    return F.cosine_similarity(reps.unsqueeze(1), reps.unsqueeze(0), dim=2)


def get_dists(reps):
    """Returns pairwise L2 distances between representations"""
    return torch.cdist(reps, reps, p=2)


def get_overlaps(reps):
    """Performs pairwise overlap calculations of SDRs"""
    reps = (reps + 1.0) / 2.0  # [-1,1] -> [0,1]
    return torch.mm(reps, reps.t())


def map_sims(sims, min_evidence, max_evidence, min_overlap, max_overlap):
    """
    *Map from similarities to overlaps using the ranges provided*

    - input range: [min_evidence, max_evidence]
    - output range: [min_overlap, max_overlap]
    """

    if min_evidence == 0 and max_evidence == 0:
        min_evidence = sims.min()
        max_evidence = sims.max()
    sims = torch.clamp(sims, min_evidence, max_evidence)

    mapped_tensor = (sims - min_evidence) / (max_evidence - min_evidence) * (
        max_overlap - min_overlap
    ) + min_overlap
    return torch.round(mapped_tensor)


def map_dists(dists, min_evidence, max_evidence, min_overlap, max_overlap):
    """
    *Map from distances to overlaps using the ranges provided*

    - input range: [min_evidence, max_evidence]
    - output range: [min_overlap, max_overlap]
    """
    return map_sims(-1 * dists, min_evidence, max_evidence, min_overlap, max_overlap)


def get_fig(sims, vmin=0, vmax=41):
    """Creates a heatmap figure from similarities"""

    fig, ax = plt.subplots()
    im = ax.imshow(
        sims.detach().cpu(), cmap="hot", interpolation="nearest", vmin=vmin, vmax=vmax
    )
    fig.colorbar(im, ax=ax)
    return fig


def get_gif_lineplot(vals):
    """Creates an animation of lineplot overtime"""

    sns.set(style="dark", context="talk")
    fig, ax = plt.subplots()

    def animate(i):
        ax.clear()
        sns.lineplot(vals[:i])
        ax.set_xlabel("epochs")
        ax.set_ylabel("Normalized Overlap Error")
        ax.legend("").set_visible(False)
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 1.0])

    return animation.FuncAnimation(fig, animate, frames=len(vals))


def get_gif_linedot(vals):
    """Creates an animation of a dot on a rendered lineplot"""

    sns.set(style="dark", context="talk")
    fig, ax = plt.subplots()

    x = list(range(0, len(vals) * 10, 10))  # customize number of points and steps
    y = vals.numpy().tolist()

    sns.lineplot(x=x, y=y)
    (dot,) = ax.plot([], [], "ro")

    # customize labels
    # TODO: Should be moved to outside the function
    ax.set_xlabel("epochs")
    ax.set_ylabel("Normalized Overlap Error")

    def animate(i):
        dot.set_data([x[i]], [y[i]])
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 1.0])
        return (dot,)

    return animation.FuncAnimation(fig, animate, frames=len(vals))


def get_gif(sims, vmin=0, vmax=41):
    """
    Creates an animation of the similarities as a heatmap.
    Also adds a counter, or a progress bar
    """

    fig, ax = plt.subplots()
    cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    im = ax.imshow(
        sims[0].detach().cpu(),
        cmap="hot",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(im, cax=cax)
    prog_bar = ax.text(
        0.5,
        1.05,
        "0/{0}".format(len(sims)),
        ha="center",
        va="center",
        fontsize=20,
        transform=ax.transAxes,
    )

    def init():
        ax.set_xticks([])
        ax.set_yticks([])
        return im, prog_bar

    def animate(i):
        im.set_data(sims[i].detach().cpu())

        prog_bar.set_text("{0}/{1}".format(i + 1, len(sims)))
        return im, prog_bar

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=len(sims),
        init_func=init,
        blit=True,
    )
    plt.close()
    return ani
