import matplotlib.pyplot as plt
import numpy as np

TBP_COLORS = {
    "blue": "#00a0df",
    "purple": "#5d11bf",
    "yellow": "#ffbe31",
}

# Data from the image
monty_training = 219050192483  # AWS total training on DMC's pretrain_dist_agent_1lm
vit_training = 79360390556727 * 200  # AWS training, ViT-b16

# Set up the data for plotting
models = ["Monty", "ViT (best)"]
training_flops = [monty_training, vit_training]

# Set up positions for bars
x = np.arange(len(models))

# Common styling function
def apply_basic_style(ax, xlabel=None, ylabel=None):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)

# 1. Vertical plot (log scale)
fig_vert_log, ax_vert_log = plt.subplots(figsize=(5, 3))
rects_vert_log = ax_vert_log.bar(x, training_flops, width=0.5, color=TBP_COLORS["blue"])
ax_vert_log.set_yscale("log")
apply_basic_style(ax_vert_log, ylabel="Training FLOPs (log scale)")
ax_vert_log.set_xticks(x)
ax_vert_log.set_xticklabels(models)
plt.tight_layout()
plt.savefig(
    "/Users/hlee/tbp/results/dmc/results/floppy/figures/fig7_flops_comparison_vertical_log.png",
    dpi=300,
)

# 2. Vertical plot (linear scale)
fig_vert_linear, ax_vert_linear = plt.subplots(figsize=(5, 3))
rects_vert_linear = ax_vert_linear.bar(
    x, training_flops, width=0.5, color=TBP_COLORS["blue"]
)
apply_basic_style(ax_vert_linear, ylabel="Training FLOPs")
ax_vert_linear.set_xticks(x)
ax_vert_linear.set_xticklabels(models)
plt.tight_layout()
plt.savefig(
    "/Users/hlee/tbp/results/dmc/results/floppy/figures/fig7_flops_comparison_vertical_linear.png",
    dpi=300,
)

# 3. Horizontal plot (log scale)
fig_horiz_log, ax_horiz_log = plt.subplots(figsize=(5, 3))
rects_horiz_log = ax_horiz_log.barh(
    x, training_flops, height=0.5, color=TBP_COLORS["blue"]
)
ax_horiz_log.set_xscale("log")
apply_basic_style(ax_horiz_log, xlabel="Training FLOPs (log scale)")
ax_horiz_log.set_yticks(x)
ax_horiz_log.set_yticklabels(models)
plt.tight_layout()
plt.savefig(
    "/Users/hlee/tbp/results/dmc/results/floppy/figures/fig7_flops_comparison_horizontal_log.png",
    dpi=300,
)

# 4. Horizontal plot (linear scale)
fig_horiz_linear, ax_horiz_linear = plt.subplots(figsize=(5, 3))
rects_horiz_linear = ax_horiz_linear.barh(
    x, training_flops, height=0.5, color=TBP_COLORS["blue"]
)
apply_basic_style(ax_horiz_linear, xlabel="Training FLOPs")
ax_horiz_linear.set_yticks(x)
ax_horiz_linear.set_yticklabels(models)
plt.tight_layout()
plt.savefig(
    "/Users/hlee/tbp/results/dmc/results/floppy/figures/fig7_flops_comparison_horizontal_linear.png",
    dpi=300,
)

plt.show()
