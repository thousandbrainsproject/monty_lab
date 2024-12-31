# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import matplotlib.pyplot as plt
import numpy as np
from IPython.core.pylabtools import figsize
from matplotlib import animation

from tbp.monty.frameworks.models.states import State


def create_animation(t, states, filename=None):
    """Animates the dynamical system and optionally saves it to file.
    It is concerned with plotting, not computing the dynamics.

    :param t: time vector.
    :param states: Nested list of State class instances (representing a
    full simulation run).
    :param length: nominal length of the chain links.
    :param filename: if provided, a movie file will be saved of the animation.
    This may take some time.
    :returns: fig: matplotlib.Figure
    :returns: anim: Animation. matplotlib.FuncAnimation
    """
    # number of objects
    n_objects = len(states[0])

    # create the axes
    fig = plt.figure()
    ax = plt.axes(
        xlim=(-1.2, 1.2),
        ylim=(-1.2, 1.2),
        zlim=(-1.2, 1.2),
        aspect="equal",
        projection="3d",
    )

    # display the current time
    time_text = ax.text(0.04, 0.1, 0.1, "", transform=ax.transAxes)

    # blank line for the pendulum & blank vectors for reference frame visualization
    (line,) = ax.plot(
        [], [], lw=2, marker="o", markersize=6, color="b", mfc="k", mec="k"
    )
    line.set_3d_properties([])

    ref_frames_ax = []
    for _ in range(3 * n_objects):
        ref_frames_ax.append(ax.quiver([], [], [], [], [], [], color="r", length=0.3))

    # initialization function: plot the background of each frame
    def init():
        time_text.set_text("")
        line.set_data([], [])
        line.set_3d_properties([])

        ref_frames_ax = []
        for _ in range(3 * n_objects):
            ref_frames_ax.append(
                ax.quiver([], [], [], [], [], [], color="r", length=0.3)
            )

    # animation function: update the objects
    def animate(i):
        time_text.set_text("time = {:2.2f}".format(t[i]))
        pos = np.zeros((n_objects, 3))
        for j in range(n_objects):
            pos[j, :] = states[i][j].location
            pose_vectors = states[i][j].morphological_features["pose_vectors"]
            for k in range(3):
                ref_frames_ax[3 * j + k].remove()
                ref_frames_ax[3 * j + k] = ax.quiver(
                    pos[j, 0],
                    pos[j, 1],
                    pos[j, 2],
                    pose_vectors[0, k],
                    pose_vectors[1, k],
                    pose_vectors[2, k],
                    color="r",
                    length=0.3,
                )
        line.set_data(pos[:, 0], pos[:, 1])
        line.set_3d_properties(pos[:, 2])

    # call the animator function
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(t),
        init_func=init,
        interval=(t[1] - t[0]) * 1000,
        blit=False,
        repeat=True,
    )
    plt.show()

    # save the animation if a filename is given
    if filename is not None:
        anim.save(filename, fps=30, codec="libx264")


def create_state_class_instance(
    location, pose_vectors, linear_velocity, angular_velocity
):
    """Returns a State class instance with given location, pose vectors and velocities.

    :param location: 3D location of the corresponding object.
    :param pose_vectors: 3 x 3 array with each column being the basis vectors
    of the object's reference frame.
    :param linear_velocity: linear velocities along axes of global reference frame.
    :param angular_velocity: angular velocity vector.
    :return: State class object with corresponding location and pose vectors.
    """
    obj_state = State(
        location=location,
        morphological_features=dict(
            pose_vectors=pose_vectors,
            pose_fully_defined=True
        ),
        non_morphological_features=dict(
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity
        ),
        confidence=1,
        use_state=True,
        sender_id="sim_environment",
        sender_type="SM",
    )

    return obj_state


def plot_state_var(t, x, dynamic=None):
    # Plot the dynamic variables vs time
    figsize(8.0, 6.0)
    plt.plot(t, x[:, :x.shape[1] // 2])
    plt.xlabel("Time [sec]")
    if dynamic:
        plt.legend(dynamic[:x.shape[1] // 2])
    plt.show()
