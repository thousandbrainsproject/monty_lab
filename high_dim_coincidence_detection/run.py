# Copyright 2025 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import json
import math
import os
import pprint
import random

import hdcd_module as hdcd
import numpy as np
import utils

if __name__ == "__main__":

    # Define hyper-parameters shared by the experiment, the environment, and
    # the learning module
    exp_params = dict(
        # === ENVIRONMENT/EXPERIMENT PARAMETERS ===
        NUM_SEEDS=1,  # Number of random seeds to average results over
        NUM_OBJECTS=[4],  # Number of objects to create and evaluate; expects a
        # list, so if you only want to do a single experiment (e.g. four
        # objects, then set to [4])
        NUM_PRESET_FEATURES=7,  # None, or a positive integer; if None,
        # randomly initialize the SDR for every feature on an object, equivalent to
        # NUM_PRESET_FEATURES = NUM_F_PER_OBJECT * NUM_OBJECTS; otherwise,
        # create n pre-set feature SDRs that are re-used (i.e. common) across
        # all synthetically generated objects; smaller n is more challenging,
        # as feature identity carries less information about object identity
        NUM_F_PER_OBJECT=30,  # Number of features created per object; because we
        # will learn on all features, this is equivalent to the number of features
        # we will visit at learning
        NUM_F_FOR_SENSING=30,  # The number of features the system
        # will actually experience at inference time; at a conceptual level,
        # this would either be due to multiple sensations, or due to inputs
        # from multiple columns
        THETA_NOISE=0.25,  # 0.25 reasonable choice for testing
        # generalization; noise (standard deviation of radians) to the
        # orientation of features relative to a "novel" object example vs. their
        # orientation on the learned object
        TRANSLATION_NOISE=0.5,  # 0.5 reasonable choice for generalization; as
        # above but for translations; objects are synthetic, so it is arbitrary
        # "distance" units, however features are initialized in a cube of dimension
        # -5:5 around the centre of each object
        EULER_MATCH_ETA=math.pi / 8,  # Radians, recommend math.pi/8; amount
        # of tolerance in the estimated pose for describing an output as correct
        TRANSLATION_MATCH_ETA=1.0,  # Recommend 0.5 or 1.0; as above
        # TODO if we return to this work, add non-binary measures of pose error
        RANDOM_ROTATIONS_MAG=2 * math.pi,  # E.g. 2*math.pi; after learning,
        # randomly rotate the objects in the environment, bounded by this
        # magnitude; used for assessing rotation invariance
        RANDOM_TRANSLATIONS_MAG=5,  # E.g. 5; as above, but for translations

        # === LEARNING MODULE PARAMETERS ===
        MINICOLUMN_DIM=256,  # E.g. 256; number of neurons per minicolumn
        SDR_SENSATION_THRESHOLD=16,  # e.g. 14; the required match between an
        # input feature and the learned alignment weights for that feature
        # to be considered by the learning module; if MINICOLUMN_DIM is high,
        # the threshold can be quite robust even when relatively low; the maximum
        # possible value is 16, because the minicolumns correspond to a 4x4
        # transformation matrix, and each minicolumn can have one winning neuron
        SDR_CLASSIFIER_THRESHOLD=6,  # e.g. 6; the required match between the
        # output SDR of an LM and an object's SDR for the LM to claim that
        # it is representing that particular class; thus with a low threshold
        # the LM might conclude it's SDR is consistent with multiple objects
        # Too high a threshold means the LM is sensitive to any deviation of the
        # SDR representation from the one formed at learning; as above, this should not
        # be set higher than 16 given that each minicolumn can have one winning neuron
        # Note this threshold does not need to scale with MINICOLUMN_DIM

        # === CLUSTERING PARAMETERS ===
        # Parameters that determine how a learning module forms a consensus
        # about its representation
        CLUSTERING_METHOD="dbscan",
        # Options are: dynamic_routing, dbscan, or hybrid_clustering; recommend
        # dbscan, although note that can be slow with objects with many features

        # = Parameters for dynamic routing =
        DYNAMIC_R_ITERS=5,  # How many iterations of dynamic routing to
        # perform; 3-5 is usually sufficient
        SOFTMAX_SCALING=20,  # How much to scale softmax operation, pushing
        # "good" and "bad" predictions farther apart; if it's too small,
        # dynamic routing may not converge, while it may converge to a bad
        # local minimum if it is too large; 10 is a reasonable starting value
        VISUALIZE_CUMM_DISTRIBUTION=False,  # Bool; whether to plot the
        # cumm. distribution of weighted predictions to the current consensus
        # in dynamic routing; useful for debugging

        # = Parameters for DBSCAN =
        BINARY_SEARCH_BOOL=True,  # Whether to perform a binary search over
        # the window radius used in DBSCAN; improves results, but slower (esp.
        # if many predictions that need clustering)
        WINDOW_SIZE=0.25,  # The *initial* window size that DBSCAN attempts to use for
        # clustering, unless not performing a binary search; with a binary search,
        # this will be dynamically modulated if we either identify no clusters, or
        # identify too many
        BINARY_ITERATIONS=20,  # Max number of iterations of the binary search
        # (and initial exponential search) over window-size; after this point is
        # reached, we declare failure (e.g. may be on a novel object)

        # == Parameters for hybrid clustering ==
        PROPORTION_FOR_EARLY_RETURN=None  # If using hybrid clustering, what
        # top-k proprtion of predictions dynamic routing should return; should
        # be bound 0:1 (e.g. 0.5), or if early returning not desired, set to None
    )

    assert (exp_params["NUM_F_PER_OBJECT"]
            >= exp_params["NUM_F_FOR_SENSING"]), "Fix # sensations"

    # Accumulate results
    results_dic = dict(
        acc_vs_num_objects=[],
        wrong_conv_vs_num_objects=[],
        no_match_vs_num_objects=[],
        pose_recovery_vs_num_objects=[],
        total_time_vs_num_objects=[],
        sense_time_vs_num_objects=[],
        pose_time_vs_num_objects=[],
    )

    print("\n===Config Overview===\n")
    pprint.pprint(exp_params)

    for current_num_objects in exp_params["NUM_OBJECTS"]:

        print(f"\n\n\nTesting recognition with {current_num_objects} objects")

        temp_acc = []
        temp_wrong_conv = []
        temp_no_match = []
        temp_pose_recovery = []
        temp_time_total = []
        temp_time_sense = []
        temp_time_pose = []

        for current_seed in range(exp_params["NUM_SEEDS"]):

            print(f"\nOn seed : {current_seed}")

            random.seed(current_seed)
            np.random.seed(current_seed)
            # Seeds are used here for creating pre-set features
            # They are re-set for the main loop to keep behaviour consistent
            # regardless of whether pre-set features are used or not

            if exp_params["NUM_PRESET_FEATURES"] is not None:
                print(f"\nUsing {exp_params['NUM_PRESET_FEATURES']} common features for"
                      " objects")
                # Generate pre-set features, i.e. a fixed set of features that
                # will be shared by objects, otherwise features will be randomly
                # generated on the fly for each object, and therefore be largely
                # unique if the dimensionality of the mini-columns is high
                preset_feature_indices = []

                for __ in range(exp_params["NUM_PRESET_FEATURES"]):

                    feature_indices = []

                    for __ in range(4 * 4):

                        # Note that at the moment, there is no explicit notion here of
                        # similarity between different SDR features (i.e. other than
                        # what happens by chance), however this is something that SDRs
                        # are capable of, and is a TODO if we return to this
                        feature_indices.append(np.random.randint(
                            0, exp_params["MINICOLUMN_DIM"]))

                    preset_feature_indices.append(feature_indices)

            else:
                print("\nUsing randomly initialized SDR for every feature")
                preset_feature_indices = None

            # Run main simulation
            current_results_dic = hdcd.simulate_hdcd_module_and_inference(
                exp_params,
                current_seed,
                current_num_objects,
                preset_feature_indices)

            temp_acc.append(
                current_results_dic["correctly_converged"] / current_num_objects)
            temp_wrong_conv.append(
                current_results_dic["wrong_converged"] / current_num_objects)
            temp_no_match.append(current_results_dic["no_match"] / current_num_objects)
            temp_pose_recovery.append(
                current_results_dic["pose_recovered"] / current_num_objects)
            temp_time_total.append(current_results_dic["time_for_inference"])
            temp_time_sense.append(current_results_dic["time_for_sensations"])
            temp_time_pose.append(current_results_dic["time_for_pose_estimate"])

        results_dic["acc_vs_num_objects"].append(np.mean(temp_acc))
        results_dic["wrong_conv_vs_num_objects"].append(np.mean(temp_wrong_conv))
        results_dic["no_match_vs_num_objects"].append(np.mean(temp_no_match))
        results_dic["pose_recovery_vs_num_objects"].append(np.mean(temp_pose_recovery))
        results_dic["total_time_vs_num_objects"].append(np.mean(temp_time_total))
        results_dic["sense_time_vs_num_objects"].append(np.mean(temp_time_sense))
        results_dic["pose_time_vs_num_objects"].append(np.mean(temp_time_pose))

        base_dir = os.path.expanduser("~/tbp/results/monty/projects/hdcd_runs/")

        if os.path.exists(base_dir) is False:
            try:
                os.makedirs(base_dir)
            except OSError:
                pass

        with open(base_dir + "results.json", "w") as f:
            json.dump(results_dic, f)

    if len(exp_params["NUM_OBJECTS"]) > 1:
        utils.plot_results(exp_params["NUM_OBJECTS"], base_dir)
