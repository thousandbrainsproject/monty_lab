# High-Dimensional Coincidence Detection (HDCD) for Object Recognition

## Structure

The `.py` files in this projects folder contain everything you need to create a simple environment filled with 3D synthetic objects, and evaluate a "high-dimensional coincidence detection" (HDCD) classifier on them. 

## Algorithm summary

HDCD is an effort at combining elements of sparse-distributed representations (SDRs) with the attributes necessary to enable pose-based computations. At its core, any "object" (which could be a feature from a lower-down learning module in a hierarchy, or a sensory input feature) is represented by a large, sparse array. Which indices are "on" in this array encode feature identity (ID), while the numerical value of these "on" values indicates the pose. The biological analogy is that ON indicates that a particular neuron in a minicolumn has spiked, while the information it is giving about pose is encoded in its spike time (i.e. the moment at which the spike actually occurred relative to some e.g. background oscillations). Thus, these representations can be viewed as "phasic-SDRs".

In our setup of rigid-body transformations in 3D space, these high-dimensional vectors can be thought of as a combination of mini-column vectors, where each minicolumn vector is dedicated to a particular element of a standard 4x4 transformation matrix. Thus if we have 20 neurons in each of our minicolumns, we will have 320 neurons, in total, dedicated to representing both the pose and ID of an object. At present, only one neuron in any given minicolumn will be selected to represent that pose element and index of the SDR.

To perform object recognition and pose prediction, we process input features, using learned weight operations to predict all the output IDs and poses (jointly) that might exist given that input feature. Given several input features from an object, there will be many diverse predictions. Importantly, predictions common to a given object are likely to cluster in the high-dimensional ID-pose space, and so by identifying these clusters, we can infer an object and its ID. Due to the high-dimensionality, this clustering is still likely to be much better for the true object than for other objects, even in the presence of noise. In order to be able to perform this operation, we also learn the necessary weights, which is made more simple by treating input objects as being in their identity pose during learning.

## Overview and Tutorial

To run an experiment, simply specify the desired hyperparameters in the `run.py` exp_params config, and then run `python run.py`. The significance of each hyperparameter is explained in `run.py`.

The necessary synthetic objects will be (quite quickly) generated, before the model switches to training and then inference. If desired, the objects will also be transformed after training but before inference, so as to make the task a bit more interesting!

### A note on the experimental setup

Note that in the current setup, the training phase has a supervised signal in that the learning module knows it is on a new object, and furthermore it knows when it transitions to a different object, either at learning or at inference. In addition, note that we receive a set of sensations (determined by `NUM_F_FOR_SENSING`) before attempting to find consensus on what the current input is, which contrasts with e.g. graph-matching. This set of sensations could be a result of either multiple steps of a single sensory module, or simultaneous input from multiple sensory modules; any difference of these two scenarios is abstracted away in the current setup. 

### A note on clustering and the phasic-SDRs

Note that when clustering, although a pose value of 0 (e.g. sin(0)), encoded by a spike (binary value 1), cannot be distinguished directly from a "pose" value of 0 (due to no spike), the pose representations are orthonormal, and thus as a particular e.g. sin(theta) approaches 0, the corresponding cos(theta) will approach $\pm1$. As clustering occurs across all elements jointly, the potential issue of predictions not being distinguishable is therefore mitigated. However, a particular SDR element will not make a significant contribution to clustering if it is associated with a pose of 0, although when determing the final SDR, we are still able to recover a winning/spiking neuron for every minicolumn by using the alignment weights formed at learning. The more significant exception to mitigating this issue is translation, which indeed appears to suffer more from noise because the location vector is not constrained to be orthonormal; a future effort would be to convert the translations into a phase code.