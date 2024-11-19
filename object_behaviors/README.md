# **A dynamic environment to simulate object behaviors**

This repository contains code implementing the simulation environment used to model object behaviors. The core of the environment is contained in the `environment.py` module. Classes instantiating specific interaction types are defined in the `interactions.py` module. Additional utilities (mostly used for plotting/animation) are given in the `utils.py` module. 

## **Run a simulation and use of the `.step()` method**

A full-length simulation can be run in the following way:

```
# Create an environment class instance & run a simulation
env = ObjectBehaviorEnvironment()
env.run_simulation()
```

In practical cases, the environment will only be queried through its `.step()` method (its only public interface). The method runs a single step of the simulation and returns the current state of the object as a list of `State` class instances (one for each object part). The reader is referred to the `.run_simulation()` method code for an example of how to use the `.step()` method.

## **Specifying the dynamical model**

### **Initial conditions**

In the current implementation, only the number of nodes (object parts) and generic simulation parameters (e.g. length of the chain links, mass of each node, gravity constant) are specified as inputs to the environment. Initial conditions are defined in terms of the dynamical variables representing the dynamics of the object, not in terms of the location and pose of the connecting nodes. Those can be specified when instantiating a particular interaction type. 

### **Interaction graph**

As for interactions, they are sampled independently from the pool of possible interaction types. They are arranged in a chain such that each part is connected to two other parts at most. While this greatly facilitates the definition of the model, it certainly does not capture the breadth of object dynamics we may want to simulate (e.g. object with cyclic interaction graphs). Note that in principle it is possible to modify the code to define cyclical interactions. This would primarily require changing the code to specify initial conditions based on a part's relative location (from which joint angles can be derived), rather than the relative angle being the initializing parameter.

Currently two major interaction classes are implemented in the `interaction.py` module: a linear (also called prismatic) joint and a planar rotational joint. On top of defining the joints (which only constrains the degrees of freedom of the system), force elements can be added (e.g. springs, damper) which can also contribute to generating more complex dynamics. At the moment, we cannot specify a standard "rigid" interaction between two points; a stiff spring in a linear joint might be a reasonable approximation, but could be numerically unstable.

### **Next steps**

In the future, one may want to specify explicitly the location and pose of all nodes as initial conditions, along with the interaction graph (set of all wanted interactions); potentially as input to the environment. This should enable greater flexibility in the model definition, although it might be more complex to set-up from a software architecture perspective. Generally speaking, model definition and derivations of EOMs is the most tricky part to get working correctly. If this part is done right, then the numerical integration should be fairly straightforward.

### **Sympy framework**

While there are many different ways of instantiating a valid dynamical model with sympy (see custom examples `stapler.py` and `pendulum.py`), the [joints framework](https://docs.sympy.org/latest/modules/physics/mechanics/joints.html) provided by `sympy.physics.mechanics` seems to be the most convenient to work with. It makes it easy to define new interaction types and to extract EOMs through the `JointsMethod`. The following examples may be useful to understand how to use the framework:

- Example of a [system with highly-heterogeneous set of interactions](https://docs.sympy.org/latest/modules/physics/mechanics/examples/multi_degree_freedom_holonomic_system.html).

- Example of a [system with cyclic interactions](https://docs.sympy.org/latest/modules/physics/mechanics/examples/four_bar_linkage_example.html).

## **Interface with a relational inference module**

The only concern of the environment is to output the state of the object when queried. The object's state is the concatenate of the state of each of its consituents (i.e. object parts). The state of each part contains at minimum the following 4 elements, all summarized in a `State` class instance: 3D location, pose vectors (both expressed w.r.t. global reference frame), linear velocity and angular velocity vector.

In addition to the state of the object, another attribute of interest is the ground-truth interaction/edge matrix, accessible through the `.get_edge_matrix()` method. The interaction matrix is an (N x N) matrix, with N the number of object parts. Each matrix entry contains the label (i.e. interaction type) of the directed edge connecting the two corresponding object parts (represented by the row and columns indices of that entry). This interaction matrix is sampled once at the beginning of every simulation run, and remains static over the course of that run. It may serve useful to test/evaluate the performance of the relational inference algorithm (check how inferred edge types differ from the ground-truth interactions).

Most relational inference algorithms take as input a temporal window of object states (not a single time step as given by the output of the environment). For this reason, a buffer (short-term memory) should be implemented in conjunction to the relational inference module. The buffer would take environment outputs from a single time step, aggregate them across time, and feed the aggregated data into the relational inference module.

## **Implementation routes for the relational inference model**

### **Approaches based on graph message passing**

Here is a set of papers that introduce relational inference algorithms based on graph neural networks (message passing), with a focus on improving performance on trajectory forecasting tasks:

* [Kipf, T., Fetaya, E., Wang, K. C., Welling, M., & Zemel, R. (2018, July). Neural relational inference for interacting systems. In International conference on machine learning (pp. 2688-2697). PMLR.](https://arxiv.org/pdf/1802.04687.pdf)

**Summary:** Learns to infer a static set of interaction edges valid for the whole duration of a simulation (i.e. assumes that the interaction between nodes remain the same throughout the simulation). Those inferred interactions/edge types are then used to condition the decoder, which is a dynamics predictor (equivalent to a state transition function). The whole model is formalized as an autoencoder, which enables end-to-end training in a self-supervised way.

* [Graber, C., & Schwing, A. G. (2020). Dynamic neural relational inference. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 8513-8522).](https://openaccess.thecvf.com/content_CVPR_2020/papers/Graber_Dynamic_Neural_Relational_Inference_CVPR_2020_paper.pdf)

**Summary:** Extends the NRI paper to learn a separate interaction graph at each time step. Works well for dynamical systems where interactions between entities may change over time. Decoder is made more flexible than the original NRI, by not assuming that the latent factors z (representing interaction types) remain the same over the simulation time course. Shows improved performance over standard NRI on a couple of benchmarks.

* [Stanić, A., van Steenkiste, S., & Schmidhuber, J. (2021, May). Hierarchical relational inference. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 35, No. 11, pp. 9730-9738).](https://arxiv.org/abs/2010.03635#:~:text=Common%2Dsense%20physical%20reasoning%20in,the%20complex%20behaviors%20they%20support.)

**Summary:** Extends the dynamical NRI paper to dynamical systems with hierarchical relationships. The use of a slot-based encoder enables local tracking of objects in the image space and removes the assumption of a fixed number of nodes in the system. As objects may switch slots throughout the simulation, a static interaction graph can not be assumed. A finite time window of past object states is used in the decoder to infer the next state.

### **Translation & rotation-invariant relational inference using pair-point features (PPF)**

All approaches above operate on node-based object state representations (used as input and output of the system), which are not invariant to translation and rotations. Also, nodes used are non-oriented nodes (no frames attached). A potential improvement over this would be to use an edge-based object state representation based on point-pair features. This representation would be pose-invariant and would therefore remove the need for observing the object in all of its possible poses. The state transitions would be formalized as a message-passing step at the level of single interactions, and not overall objects. This should enable greater generalization performance on unseen objects through composition of known interaction types. Those transitions may be conditioned on agent-generated actions and/or object states.

This is a preliminary idea currently under investigation. Additional details and clarification will be added as the model architecture is being worked out.

### **General specifications**

Beyond trajectory forecasting (next-state prediction), here is a list of general specifications we want our object modeling framework to address:

### *Learning:*
* Recognize how change of one feature-pose point in the graph affects other points in the graph -> identify graph edges & their types (what is their influence on other nodes?).
* Represent a simple sequence of states as a behavior. We also want the ability to learn action-conditioned sequences (i.e. understand how action affect transitions between object states).

### *Inference:*
* Recognize an object in a given state (not moving) from observing a subset of its features.
* Given we recognized the objects pose and state, predict features that we will see on unobserved locations on the object.
* Infer how the objects state could change from the current state.
* Recognize a behavior from a sequence of states.
* Predict the next state of an object, given its current state (including forces being applied) and behavior -> Usually next state can be predicted just from knowing the forces but forces are not always known (like when observing someone else moving).

### **Open questions:**

Here is a list of open questions which would be useful to get further clarification on:

* Where in the Monty system would information for state & behavior recognition be communicated? Lateral votes? Large fan-in input from multiple lower-level vote population?
* Is there a way we can avoid storing a model for each object state (associated with same object ID)? For example by storing node relationships & then applying them to a default graph to transform it into different states (i.e. transfer learning).