# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import copy as cp

import numpy as np
import sympy as sp
import sympy.physics.mechanics as me
from scipy.integrate import odeint

from interactions import LinearJoint, PlanarJoint
from utils import create_animation, create_state_class_instance


class ObjectBehaviorEnvironment:
    def __init__(self, n_parts=3, link_length=0.5, mass=0.01, gravity=9.81):
        """Class implementing a dynamic environment to simulate object behaviors.
        The typical set-up consists of a set of object parts (modelled as point masses
        with attached reference frames) arranged in a chain. Each link instantiates
        a particular interaction type between the two connected objects (sampled at
        the beginning of each simulation). One end of the chain is fixed, while the
        other end is freely moving and can be influenced by agent-generated actions
        (3D forces). Note that the current environment implementation does not
        allow for defining interaction loops (where one node may connect to more than
        two other nodes). Interactions are defined as separate classes in the the
        interactions.py module which gives flexibility to define novel interaction
        types in the future.

        :param n_parts: number of objects in the simulation.
        :param link_length: length of each static link i.e. distance between two
        object parts (not applicable to linear springs). All links in the system
        share the same link length.
        :param mass: mass of each object part. All object parts share the same
        mass in the system.
        :param gravity: amount of gravity to apply to the simulation. Default is the
        gravity on earth.
        """
        # Initialize class attributes
        self.n_parts = n_parts
        self.link_length = link_length
        self.mass = mass
        self.gravity = gravity

        # Set up dynamical model and derive symbolic equations of motion (EOMs)
        self._setup_environment()

        # Set initial conditions for the simulation
        r_init, v_init = [], []
        for i in range(len(self.edges)):
            edge_i = self.edges[i]
            r_init_i, v_init_i = edge_i.get_initial_conditions()
            r_init.append(r_init_i)
            v_init.append(v_init_i)

        # Array containing the current values of the dynamical variables that
        # describe the dynamics of the object. This variable gets updated in
        # the .step() method.
        self.x = np.array(r_init + v_init)

        # List of state class instances that represent the initial state of the system.
        self.init_state = self.get_state()

    def _setup_environment(self):
        """Defines a dynamical model (samples interactions between object parts) and
        derives corresponding symbolic equations of motions (EOMs). Equations of motion
        are differential equations defined on the set of all dynamical variables that,
        together, fully-determine the state of the system at any point in time.

        In practice, dynamical variables are introduced for each sampled interaction.
        For a planar rotational joint the relevant dynamical variable would be the angle
        of the joint and its angular velocity, while for a linear joint or spring it
        would be the displacement in the linear direction of the joint and its linear
        velocity. Note that those dynamical variables always come in pair, which we will
        refer to as "r" and "v" to be consistent with most online sympy examples. "r" is
        the variable that defines the current state of the joint at time t (e.g. angle
        of a planar rotational joint), while "v" is its time derivative (e.g. angular
        velocity). For joints with more than one degree of freedom (e.g. spherical
        joint), there will be one "r" and one "v" per DOF. The .get_dynamic_variables()
        method of that interaction type should then output the list of all "r" variables
        and "v" variables for r_i and v_i respectively.

        Both variables need to be solved for (through the EOMs) in order to simulate
        the dynamics of the system. Once determined, they can be mapped to the absolute
        3D coordinates, pose vectors, velocities, etc... of each object part.
        """
        # Define generic simulation parameters (link length, mass, gravity). One
        # dictionary is used to collect symbols for each parameter, another one is used
        # to collect their respective values.
        length_symbol, mass_symbol, gravity_symbol = sp.symbols("l, m, g")
        generic_parameter_symbols = dict(
            l=length_symbol, m=mass_symbol, g=gravity_symbol
        )
        generic_parameter_values = dict(l=self.link_length, m=self.mass, g=self.gravity)

        # Define possible interactions. Note that the label count start at 1 since
        # 0 should represent "no interaction" in the interaction matrix.
        edge_types = [LinearJoint, PlanarJoint]
        edge_labels = np.arange(1, len(edge_types) + 1)

        # Define a body object for the fixed base. Bodies are sympy objects (should be
        # one for each object part)
        base = me.Body("base")

        # Define list containers for all system variables
        r, v = [], []
        bodies, joints, self.edges = [base], [], []
        self.edge_matrix = np.zeros((self.n_parts, self.n_parts))

        # Collects all static parameters of the system (both generic and
        # interaction-specific) in two separate dictionaries: one for sympy symbols,
        # one for their values. Dictionaries are used instead of lists for robustness
        # against parameter ordering.
        self.parameter_symbols = cp.deepcopy(generic_parameter_symbols)
        self.parameter_values = cp.deepcopy(generic_parameter_values)

        # Define a sympy-compliant model of the system
        for i in range(1, self.n_parts):
            # Define a body instance for current object part
            body_i = me.Body("B" + str(i), mass=mass_symbol)

            # Sample interaction type
            edge_label_i = np.random.choice(edge_labels)
            edge_type_i = edge_types[edge_label_i - 1]

            # Update edge/interaction matrix
            self.edge_matrix[i - 1, i] = edge_label_i
            self.edge_matrix[i, i - 1] = edge_label_i

            # Defines an edge_type class instance. Contains all interaction-specific
            # computations and attributes. Also extract relevant dynamical variables
            # and parameters.
            edge_i = edge_type_i(
                node_id=i,
                parent=bodies[-1],
                child=body_i,
                world=base,
                generic_parameter_symbols=generic_parameter_symbols,
            )

            # Get the dynamic variables characterizing the current edge
            r_i, v_i = edge_i.get_dynamic_variables()

            # Get the sympy Joint class instance of the current edge
            joint_i = edge_i.get_joint()

            # Get the sympy symbols representing the parameters of each joint and their
            # corresponding values
            (
                parameter_symbols_i,
                parameter_values_i,
            ) = edge_i.get_interaction_parameters()

            # Append data to lists
            r += r_i
            v += v_i
            self.edges.append(edge_i)
            bodies.append(body_i)
            joints.append(joint_i)

            # Update dictionaries with interaction-specific parameters
            self.parameter_symbols.update(parameter_symbols_i)
            self.parameter_values.update(parameter_values_i)

        dynamic_variables = r + v

        # Compute EOMs in symbolic form using Joints Method
        method = me.JointsMethod(bodies[0], *joints)
        method.form_eoms()

        # Convert symbolic EOMs into numerically evaluable equations
        # Note: list(self.parameter_symbols.values()) outputs a list of sympy symbols
        # corresponding to all static parameters in the system.
        self.M_func = sp.lambdify(
            dynamic_variables + list(self.parameter_symbols.values()),
            method.mass_matrix_full,
        )
        self.f_func = sp.lambdify(
            dynamic_variables + list(self.parameter_symbols.values()),
            method.forcing_full,
        )

    def _get_derivatives(self, x, t, args):
        """Returns the time derivatives of the states by solving the following system:
        M_func * dx/dt = f_func

        M_func is the mass matrix, and f_func is the forcing vector. This function is
        used in conjunction with "odeint" in the step method.
        Further explanations for why those tensors are needed can be found here:
        https://docs.sympy.org/latest/modules/physics/mechanics/kane.html

        :param x: current values of the dynamical variables.
        :param t: current time (given in seconds).
        :param args: extra arguments needed for the computation of M_func and f_func.
        :returns: dx_dt: The derivative of the state.
        """
        # Solving for the derivatives. Output of "solve" is a (length(x), 1) matrix.
        # Indexing just removes the last dimension of the array.
        # Note: the parameters provided to the self.M_func and self.f_func functions
        # are the same parameters that were provided as arguments to the lambdify
        # function in the ._setup_environment() method.
        dx_dt = np.linalg.solve(self.M_func(*x, **args), self.f_func(*x, **args))[:, 0]

        return dx_dt

    def get_edge_matrix(self):
        """Returns the ground truth interaction matrix. Interaction types are
        represented in the matrix by integers, but these map onto specific interaction
        classes (e.g. LinearJoint).
        """
        return self.edge_matrix

    def get_state(self):
        """Translates numerically solved dynamical variables into a list of State class
        instances.

        :returns: states: list summarizing the state of all object-parts for the current
        simulation time step. List of state class instances, one for each object part.
        """
        # Define base-part state as a State class instance
        base_part_state = create_state_class_instance(
            location=np.zeros(3),
            pose_vectors=np.eye(3),
            linear_velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
        )

        # Run through all object parts
        object_state = [base_part_state]
        for j in range(1, self.n_parts):
            # Extract current object part pose based on interaction type
            edge_i = self.edges[j - 1]
            r_i, v_i = self.x[j - 1], self.x[self.n_parts + j - 2]
            features = edge_i.get_child_pose(
                object_state[-1], r_i, v_i, self.link_length
            )

            # Summarize locations and poses in a State class instance
            current_part_state = create_state_class_instance(*features)

            object_state.append(current_part_state)
        return object_state

    def step(self, step_duration=0.01, temporal_resolution=0.01, forces=None):
        """Runs a single step of the simulation and output current system state. Actions
        can be instantaneously applied to influence the dynamics of the system. Should
        also output the connectivity of the system and a context signal to say that the
        system has changed.

        The method makes use of odeint which is responsible for computing the future
        state of the system (w.r.t. dynamical variables). odeint is a numerical solver
        provided by scipy that integrates a system of ordinary differential equations
        of the form:

        dx/dt = func(x, t, ...)

        From the definition of dx/dt and a set of initial conditions, it computes future
        x(t) values at defined timesteps stored in t.
        Here the _get_derivative method implements function "func". Attribute self.x
        keeps track of the current state of the dynamical variables and uses it at every
        step as the initial condition for the odeint function. Variable t is an array
        that defines the time stamps for which to determine the future x values for. The
        args represent additional arguments that "func" may require to compute the
        derivatives.

        :param step_duration: duration of a single simulation step (in seconds). This is
        the "frame-rate" that an observer viewing the simulation will experience.
        :param temporal_resolution: time resolution for numerical integration (in sec).
        :param forces: set of forces to apply on the object at the current timestep.
        Consists of a list of dict, with each dict specifying the id of a node to apply
        force to ("node_id" attribute), and the 3D force amounts to apply on that node.
        As such, forces can simultaneously be applied at multiple points on the object
        simply by providing multiple elements to the forces list. If forces is None, no
        force is applied at the current timestep. Per experience, force values (per
        dimension) around 20 seem to reasonably affect the dynamics.
        :return: current_state: list of state class instances summarizing current state
        of all object parts.
        """
        # Define time vector
        assert step_duration >= temporal_resolution
        n_samples = int(step_duration // temporal_resolution) + 1
        t = np.linspace(0.0, step_duration, num=n_samples)

        # Apply forces on the object
        if forces is not None:
            for node_id in forces.keys():
                force_labels = ["f_x", "f_y", "f_z"]
                force_labels = [label + str(node_id) for label in force_labels]

                force_amount = forces[node_id]
                self.parameter_values[force_labels[0]] = force_amount[0]
                self.parameter_values[force_labels[1]] = force_amount[1]
                self.parameter_values[force_labels[2]] = force_amount[2]

        # Solve EOMs through numerical integration
        x_future = odeint(
            self._get_derivatives, self.x, t, args=(self.parameter_values,)
        )
        self.x = x_future[-1, :]

        # Resets applied forces back to zero (default).
        if forces is not None:
            for node_id in forces.keys():
                force_labels = ["f_x", "f_y", "f_z"]
                force_labels = [label + str(node_id) for label in force_labels]

                self.parameter_values[force_labels[0]] = 0
                self.parameter_values[force_labels[1]] = 0
                self.parameter_values[force_labels[2]] = 0

        # Return current state
        current_state = self.get_state()

        return current_state

    def run_simulation(
        self,
        simulation_duration=3.0,
        step_duration=0.01,
        temporal_resolution=0.01,
        visualize=True,
        filename=None,
    ):
        """Runs a full simulation with the current dynamical system. Only provided
        for convenience. The .step() method should be the only public interface.
        TODO: Here I hardcoded a set-up where force is applied at step 100 on the last
        object in the chain, as a place holder. In the future, where/when forces are
        applied will have to be determined by an action policy.

        :param simulation_duration: total simulation time (in seconds).
        :param step_duration: time step for sampling the system's trajectory (different
        than the time step for numerical integration) (in seconds).
        :param temporal_resolution: time step for numerical integration (in seconds).
        :param visualize: boolean, whether or not to create an 3D animation for
        the simulation.
        :param filename: path of the file used to save the simulation. File extensions
        .mp4 or .gif typically work pretty well.
        """

        # Number of simulation steps
        n_step = int(simulation_duration // step_duration)

        # Run simulation
        sampled_trajectory, t = [self.get_state()], [0]
        for i in range(1, n_step):
            # Define a force schedule
            forces = dict()
            if i == 100:
                last_node = self.n_parts - 1
                forces[last_node] = [0, 0, -20]
            else:
                forces = None

            # get object's state
            state = self.step(
                step_duration=step_duration,
                temporal_resolution=temporal_resolution,
                forces=forces,
            )

            # Append current object state to the sampled trajectory (list of object
            # states). The time vector t records the corresponding timestamps.
            sampled_trajectory.append(state)
            t.append(t[-1] + step_duration)

        # Create visual animation for current run
        visualize = True
        if visualize:
            create_animation(t, sampled_trajectory, filename=filename)


if __name__ == "__main__":
    # Create environment object
    env = ObjectBehaviorEnvironment()

    # Run simulation
    env.run_simulation()
