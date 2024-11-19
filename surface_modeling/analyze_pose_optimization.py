# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import matplotlib.pyplot as plt
import numpy as np
from tbp.monty.frameworks.environment_utils.transforms import RandomRigidBody


def label_idx_per_class(exp, classes=None):
    """
    :param exp: data[experiment_name], e.g. data['rotation_0']
    """

    if classes is None:
        classes = np.unique(exp["labels"])

    class_to_idx = dict()
    for label in classes:
        class_to_idx[label] = np.where(exp["labels"] == label)[0]

    return class_to_idx


def get_label_indices_per_exp(data):
    """
    Get dictionary like ~
        rotation_1
            chair
                indices
            bathtub
                indices
            ...
        rotation_2
        ...

    :param data: the dict saved by online_optimization_experiment
    """

    experiments = list(data.keys())
    classes = np.unique(data[experiments[0]]["labels"])
    label_indices_per_exp = dict()
    for exp in experiments:
        label_indices_per_exp[exp] = label_idx_per_class(data[exp], classes=classes)

    return label_indices_per_exp


def get_error_per_exp(data):
    """
    Get a numpy array n_exps x n_samples where all_errors[i, j] = error exp i sample j

    :param data: nested dict, result of torch.load(.../stats.pt)
    """
    n_exps = len(data.keys())
    keys = list(data.keys())
    n_samples_per_exp = len(data[keys[0]]["mean_pointwise_error"])
    all_errors = np.zeros((n_exps, n_samples_per_exp))
    for i, k in enumerate(data.keys()):
        all_errors[i] = data[k]["mean_pointwise_error"]

    return all_errors


def get_time_per_exp(data):
    """
    Exactly like get_error_per_exp but replace error with time

    :param data: nested dict, result of torch.load(.../stats.pt)
    """
    n_exps = len(data.keys())
    keys = list(data.keys())
    n_samples_per_exp = len(data[keys[0]]["time"])
    all_times = np.zeros((n_exps, n_samples_per_exp))
    for i, k in enumerate(data.keys()):
        all_times[i] = data[k]["time"]

    return all_times


def get_euler_angles_per_transform(data):
    """
    In each experiment, extract euler angles of transform applied to src point cloud

    :param data: nested dict, result of torch.load(.../stats.pt)
    """
    angles = []
    for k in data.keys():
        if isinstance(data[k]["transforms"], RandomRigidBody):
            tsfm = data[k]["transforms"].rotation_transform.rotation
            angles.append(tsfm.as_euler("xyz", degrees=True))
        else:
            angles.append(data[k]["transforms"].rotation.as_euler("xyz", degrees=True))

    angles = np.array(angles)
    return angles


def error_per_class(data):
    """
    Group data by class label aggregating all rotations or experiments

    :param data: nested dict, result of torch.load(.../stats.pt)
    :return: class_to_error: dict[class_label] -> np.array([errors_exp_1, errors_2...])
    """
    label_indices_per_exp = get_label_indices_per_exp(data)
    exps = list(data.keys())
    classes = list(label_indices_per_exp[exps[0]].keys())
    n_samples_per_class = len(label_indices_per_exp[exps[0]][classes[0]])
    n_samples = len(exps) * n_samples_per_class
    class_to_error = {label: np.zeros(n_samples) for label in classes}

    for label in classes:
        for i, exp in enumerate(exps):
            err = data[exp]["mean_pointwise_error"][label_indices_per_exp[exp][label]]
            start = i * n_samples_per_class
            stop = (i + 1) * n_samples_per_class
            class_to_error[label][start: stop] = err

    return class_to_error


def plot_mean_error_per_class(data, exp_name=None):
    """
    Plot mean error per class label with error bars, averaging across all experiments

    :param data: the dict saved by online_optimization_experiment
    :return fig, ax: the figure data. Note, you hit plt.show()!
    """

    # Get labels, means, and variances
    class_to_error = error_per_class(data)
    classes = list(class_to_error.keys())
    error_means = np.array([class_to_error[k].mean() for k in classes])
    error_vars = np.array([class_to_error[k].var() for k in classes])

    # Sort everything by mean error
    error_sort = np.argsort(error_means)
    means_by_error = error_means[error_sort]
    vars_by_error = error_vars[error_sort]
    classes_by_error = [classes[i] for i in error_sort]

    # Make the main plot
    fig, ax = plt.subplots()
    ax.bar(x=classes_by_error, height=means_by_error, yerr=vars_by_error)
    ax.set_xticklabels(classes_by_error, rotation=70)

    # Gather data for an informative title
    n_samples_per_class = len(class_to_error[classes[0]])
    n_exps = len(data.keys())

    title = ""
    if exp_name:
        title += f"{exp_name}\n"
    title += "Error per class\n"
    title += f"(averaged over {n_samples_per_class} samples per class"
    title += f" in {n_exps} transformations)"
    ax.set_title(title)

    return fig, ax


def plot_error_vs_transform_degrees(all_errors, rots, exp_name=None):

    fig, ax = plt.subplots()
    errors = all_errors.mean(axis=1)
    evars = all_errors.var(axis=1)

    ax.errorbar(rots, errors, yerr=evars, marker=".", ls="none")
    ax.set_ylabel("Mean pointwise error")
    ax.set_xlabel("Degrees of rotation")

    if exp_name:
        ax.set_title(exp_name)

    return fig, ax


def plot_error_vs_2_transform_degrees(all_errors,
                                      rots1,
                                      rots2,
                                      label1,
                                      label2,
                                      exp_name=None):

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    errors = all_errors.mean(axis=1)
    evars = all_errors.var(axis=1)

    ax.errorbar(rots1, rots2, errors, zerr=evars, marker=".", ls="none")
    ax.set_ylabel(label1)
    ax.set_xlabel(label1)
    ax.set_zlabel("Mean pointwise error")

    if exp_name:
        ax.set_title(exp_name)

    return fig, ax


def plot_error_vs_3_transform_degrees(all_errors,
                                      rots,
                                      labels,
                                      exp_name=None):

    errors = all_errors.mean(axis=1)
    evars = all_errors.var(axis=1)

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(131, projection="3d")

    ax1.errorbar(rots[:, 0], rots[:, 1], errors, zerr=evars, marker=".", ls="none")
    ax1.set_ylabel(labels[0])
    ax1.set_xlabel(labels[1])
    ax1.set_zlabel("Mean pointwise error")

    ax2 = fig.add_subplot(132, projection="3d")
    ax2.errorbar(rots[:, 0], rots[:, 2], errors, zerr=evars, marker=".", ls="none")
    ax2.set_ylabel(labels[0])
    ax2.set_xlabel(labels[2])
    ax2.set_zlabel("Mean pointwise error")

    ax3 = fig.add_subplot(133, projection="3d")
    ax3.errorbar(rots[:, 1], rots[:, 2], errors, zerr=evars, marker=".", ls="none")
    ax3.set_ylabel(labels[1])
    ax3.set_xlabel(labels[2])
    ax3.set_zlabel("Mean pointwise error")

    axes = [ax1, ax2, ax3]
    if exp_name:
        for ax in axes:
            ax.set_title(exp_name)

    return fig, axes
