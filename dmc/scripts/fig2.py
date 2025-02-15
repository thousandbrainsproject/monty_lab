import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from data_utils import (
    DMC_ANALYSIS_DIR,
    VISUALIZATION_RESULTS_DIR,
    load_object_model,
)
from plot_utils import axes3d_clean

OUT_DIR = DMC_ANALYSIS_DIR / "fig2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


obj = load_object_model("dist_agent_1lm_10distinctobj", "potted_meat_can")
obj -= obj.translation
obj = obj.rotated(90, 260, 0)
# obj = obj.rotated(0, 0, 180)
fig = plt.figure(figsize=(2, 2))
ax = fig.add_subplot(projection="3d")
ax.scatter(obj.x, obj.y, obj.z, c=obj.rgba, marker="o", s=10, alpha=1)
axes3d_clean(ax)
ax.view_init(elev=10, azim=10, roll=0)
fig.tight_layout()
plt.show()
fig.savefig(OUT_DIR / "potted_meat_can.png", dpi=300)
fig.savefig(OUT_DIR / "potted_meat_can.pdf")
