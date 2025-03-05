import matplotlib.pyplot as plt
import numpy as np

# Data from the image
monty_inference = 3.64e10  # AWS avg inference with x_percent_threshold=20%
monty_training = 4.23e12  # AWS total training on DMC's pretrain_dist_agent_1lm
vit_inference = 3.83e10  # AWS eval on one image, ViT-b16
vit_training = 1.59e16  # AWS training, ViT-b16

# Set up the data for plotting
models = ["Monty", "ViT"]
inference_flops = [monty_inference, vit_inference]
training_flops = [monty_training, vit_training]

# Set up positions for bars
x = np.arange(len(models))
width = 0.35  # Width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(
    x - width / 2, inference_flops, width, label="Inference", color="skyblue"
)
rects2 = ax.bar(
    x + width / 2, training_flops, width, label="Training", color="lightcoral"
)

# Customize the plot
ax.set_ylabel("FLOPs (log scale)")
ax.set_title("Comparison of FLOPs: Monty vs ViT")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# Use log scale for y-axis due to large differences in values
ax.set_yscale("log")


# Add value labels on top of each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{height:.2e}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            rotation=90,
        )


autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.show()
