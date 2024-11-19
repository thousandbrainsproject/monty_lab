# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


import pickle

import matplotlib.animation as animation
import matplotlib.pyplot as plt

# Load results. If not exist, run get_icp_pointclouds.py
with open("icp_pointclouds.pkl", "rb") as f:
    result = pickle.load(f)

# Unpack some variables
n_steps = len(result["sources_good"])
dst = result["dst"]
src = result["sources_good"][0]
src2 = result["sources_bad"][0]
label = result["label"]

fig = plt.figure(figsize=(10, 5))

ax = fig.add_subplot(121, projection="3d")
graph, = ax.plot(src[:, 0], src[:, 1], src[:, 2], ".", color="g")
ax.plot(dst[:, 0], dst[:, 1], dst[:, 2], ".", color="r")
title = ax.set_title("3D Test")


ax2d = fig.add_subplot(122)

ax2d.set_ylim(-10, 200)
ax2d.set_xlim(0, n_steps)
line, = ax2d.plot([], [], "-.", linewidth=2)
ax2d.set_ylabel("Error")
ax2d.set_xlabel("Time step")


def update_graph(step, rot_srcs, errors):
    pc = rot_srcs[step]

    # graph is defined outside function scope above
    graph.set_data(pc[:, 0], pc[:, 1])
    graph.set_3d_properties(pc[:, 2])

    # title is defined outside function scope above
    title.set_text(f"Point cloud at time {step}\nLabel = {label}")
    # Rotation={angles[step]}")

    xdata = list(range(1, step + 1))
    ydata = errors[:step]
    line.set_data(xdata, ydata)

    return title, graph, line


# Heeeeeeeeeyyyyyyy Aniiiiiiiiii
ani = animation.FuncAnimation(
    fig,
    func=update_graph,
    frames=range(n_steps),
    fargs=(result["sources_good"], result["errors_good"]),
)

Writer = animation.writers["ffmpeg"]
writer = Writer(fps=20, metadata=dict(artist="Me"), bitrate=1800)
ani.save("ICP_good.mp4", writer=writer)


plt.show()


fig = plt.figure(figsize=(10, 5))

ax = fig.add_subplot(121, projection="3d")
graph, = ax.plot(src2[:, 0], src2[:, 1], src2[:, 2], ".", color="g")
ax.plot(dst[:, 0], dst[:, 1], dst[:, 2], ".", color="r")
title = ax.set_title("3D Test")


ax2d = fig.add_subplot(122)

ax2d.set_ylim(-10, 1100)
ax2d.set_xlim(0, n_steps)
line, = ax2d.plot([], [], "-.", linewidth=2)
ax2d.set_ylabel("Error")
ax2d.set_xlabel("Time step")


ani2 = animation.FuncAnimation(
    fig,
    func=update_graph,
    frames=range(n_steps),
    fargs=(result["sources_bad"], result["errors_bad"]),
)

plt.show()


ani2.save("ICP_bad.mp4", writer=writer)
