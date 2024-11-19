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

from environment import ObjectBehaviorEnvironment
from interactions import PlanarJoint
from utils import create_animation


class PendulumEnvironment(ObjectBehaviorEnvironment):
    def _setup_environment(self):
        """Defines a custom environment to simulate an arbitrarily long pendulum.
        Check the ObjectBehaviorEnvironment() class for implementation details."""
        # Define generic simulation parameters (link length, mass, gravity). One
        # dictionary is used to collect symbols for each parameter, another one is used
        # to collect their respective values.
        length_symbol, mass_symbol, gravity_symbol = sp.symbols("l, m, g")
        generic_parameter_symbols = dict(
            l=length_symbol, m=mass_symbol, g=gravity_symbol
        )
        generic_parameter_values = dict(l=self.link_length, m=self.mass, g=self.gravity)

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

            # Update edge/interaction matrix
            self.edge_matrix[i - 1, i] = 1
            self.edge_matrix[i, i - 1] = 1

            # Define an edge type object instance. Contains all joint-specific
            # computations and attributes. Also extract relevant dynamical variables
            # and parameters.
            edge_i = PlanarJoint(
                node_id=i,
                parent=bodies[-1],
                child=body_i,
                world=base,
                generic_parameter_symbols=generic_parameter_symbols,
                sample_interframe=False,
                r_init=0
            )

            # Get the dynamical variables characterizing the current edge
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
        self.M_func = sp.lambdify(
            dynamic_variables + list(self.parameter_symbols.values()),
            method.mass_matrix_full
        )
        self.f_func = sp.lambdify(
            dynamic_variables + list(self.parameter_symbols.values()),
            method.forcing_full
        )

    def run_simulation(
        self,
        simulation_duration=3.0,
        step_duration=0.01,
        temporal_resolution=0.01,
        visualize=True,
        filename=None,
    ):
        """Runs a full simulation of the pendulum."""
        # Number of simulation steps
        n_step = int(simulation_duration // step_duration)

        # Run simulation
        sampled_trajectory, t = [self.get_state()], [0]
        for _ in range(1, n_step):
            # get object's state
            state = self.step(
                step_duration=step_duration,
                temporal_resolution=temporal_resolution,
                forces=None
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
    env = PendulumEnvironment(n_parts=4)

    # Run simulation
    env.run_simulation()
