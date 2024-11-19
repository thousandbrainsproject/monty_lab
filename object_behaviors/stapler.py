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
from utils import create_animation, create_state_class_instance


class StaplerEnvironment(ObjectBehaviorEnvironment):
    def __init__(self, link_length=1.0, mass=0.01, gravity=9.81):
        # Define initial class attributes
        self.link_length = link_length
        self.mass = mass
        self.gravity = gravity

        # Setup the object model in symbolic form
        self._setup_environment()

        # Set initial conditions for the simulations
        init_cond = self.edge.get_initial_conditions()
        self.x = np.array(init_cond)

    def _setup_environment(self):
        # Define generic simulation parameters (link length, mass, gravity). One
        # dictionary is used to collect symbols for each parameter, another one is used
        # to collect their respective values.
        length_symbol, mass_symbol, gravity_symbol = sp.symbols("l, m, g")
        generic_parameter_symbols = dict(
            l=length_symbol, m=mass_symbol, g=gravity_symbol
        )
        generic_parameter_values = dict(l=self.link_length, m=self.mass, g=self.gravity)

        # Collects all static parameters of the system (both generic and
        # interaction-specific) in two separate dictionaries: one for sympy symbols,
        # one for their values. Dictionaries are used instead of lists for robustness
        # against parameter ordering.
        self.parameter_symbols = cp.deepcopy(generic_parameter_symbols)
        self.parameter_values = cp.deepcopy(generic_parameter_values)

        # Define bodies
        self.stapler_base = me.Body("stapler_base")
        self.stapler_top = me.Body("stapler_top", mass=mass_symbol)

        # Define interaction for the stapler
        self.edge = PlanarJoint(
            0,
            self.stapler_base,
            self.stapler_top,
            self.stapler_base,
            generic_parameter_symbols,
            use_spring=True,
            sample_interframe=False,
            r_init=-np.pi / 2,
        )

        # Get the dynamical variables characterizing the current edge
        r, v = self.edge.get_dynamic_variables()
        self.dynamic_variables = r + v

        # Get the sympy Joint class instance of the current edge
        stapler_joint = self.edge.get_joint()

        # Get the sympy symbols representing the parameters of each joint and their
        # corresponding values
        (
            parameter_symbols_i,
            parameter_values_i,
        ) = self.edge.get_interaction_parameters()

        # Get all parameters
        self.parameter_symbols.update(parameter_symbols_i)
        self.parameter_values.update(parameter_values_i)

        # Define Joints Method
        method = me.JointsMethod(self.stapler_base, stapler_joint)
        method.form_eoms()

        # Create a callable function to evaluate the mass matrix and forcing vector
        self.M_func = sp.lambdify(
            self.dynamic_variables + list(self.parameter_symbols.values()),
            method.mass_matrix_full
        )
        self.f_func = sp.lambdify(
            self.dynamic_variables + list(self.parameter_symbols.values()),
            method.forcing_full
        )

    def get_state(self):
        """Translates numerically solved dynamical variables into a list of State
        class instance.
        TODO: figure out a way, here or elsewhere, to adjust the object pose.

        :returns: states: list summarizing all object states for current frame. List
        of state class instances, one for each object part.
        """
        # Define base object state
        base_state = create_state_class_instance(
            location=np.array([self.link_length, 0, 0]),
            pose_vectors=np.eye(3),
            linear_velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
        )

        stapler_base_state = create_state_class_instance(
            location=np.array([0, 0, 0]),
            pose_vectors=np.eye(3),
            linear_velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
        )

        # Compute all features for stapler_top
        features = self.edge.get_child_pose(
            stapler_base_state, self.x[0], self.x[1], self.link_length
        )
        stapler_top_state = create_state_class_instance(*features)

        object_state = [base_state, stapler_base_state, stapler_top_state]
        return object_state

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
        TODO: Here I hardcoded a set-up where force is applied at step 100 as a place
        holder. In the future, where/when forces are applied will have to be determined
        by an action policy.

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
        for _ in range(1, n_step):
            # Run a single simulation step
            state = self.step(
                step_duration=step_duration,
                temporal_resolution=temporal_resolution,
                forces=None
            )

            # Append current system state to the sampled trajectory, which captures the
            # whole simulation dynamics as a list of system states. The time vector t
            # just record the time step of each sampled system states in the tajectory.
            sampled_trajectory.append(state)
            t.append(t[-1] + step_duration)

        # Create visual animation for current run
        visualize = True
        if visualize:
            create_animation(t, sampled_trajectory, filename=filename)


if __name__ == "__main__":
    # Create environment object
    env = StaplerEnvironment()

    # Run simulation
    env.run_simulation()
