# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import numpy as np
import sympy as sp
import sympy.physics.mechanics as me
from scipy.spatial.transform import Rotation as Rot


class LinearJoint:
    def __init__(
        self,
        node_id,
        parent,
        child,
        world,
        generic_parameter_symbols,
        use_spring=True,
        use_damper=False,
        use_gravity=True,
        apply_force=True,
        sample_interframe=True,
        r_init=1.0,
        v_init=1e-1,
    ):
        """Class implementing linear joint connecting two object parts (parent and
        child). It builds on sympys's PrismaticJoint class, as defined here:
        https://docs.sympy.org/latest/modules/physics/mechanics/api/joint.html

        Only has single degree of freedom in an arbitrary linear direction
        (defined as the x axis of the parent's body interframe). The parent's
        interframe is an intermediary reference frame (different from both the
        parent and child reference frames), whose orientation is sampled uniformly
        based on euler coordinates. It is used to define the linear direction
        of the joint, hence decoupling the joint direction from the parent's reference
        frame. More information on this can be found here:
        https://docs.sympy.org/latest/modules/physics/mechanics/joints.html

        While the joint itself only constrains the dynamics of the two connected bodies
        to specific areas of the state space, additional force elements can be added to
        create richer dynamics. This is the interaction to use when defining linear
        springs or dampers for example.

        :param node_id: index of child in the chain (only used to name sympy variables).
        :param parent: parent object part, body instance. Part which the edge originates
        from.
        :param child: child object part, body instance. Part which the edge directs to.
        :param world: global reference frame of the simulation
        :param generic_parameter_symbols: List of generic simulation parameters. For
        now, these are limited to the link_length, mass per node and gravity constant.
        Need to be provided as they influence the forces applied on the child object.
        :param use_spring: boolean, defines whether to add a linear spring to the joint.
        :param use_damper: boolean, defines whether to use a damping element on top of
        the joint.
        :param use_gravity: whether to add gravity to the forces acting on child object.
        :param apply_force: whether to allow applying force on child object.
        :param sample_interframe: whether to sample an interframe (i.e. arbitrary
        rotation) for the current joint.
        :param r_init: initial conditions to use for the position (dynamical variable)
        of the linear joint.
        :param v_init: initial conditions to use for the velocity (time-derivative of
        position dynamical variable) of the joint.
        """
        self.node_id = node_id
        self.parent = parent
        self.child = child
        self.world = world
        self.use_spring = use_spring
        self.use_damper = use_damper
        self.init_cond = [r_init, v_init]

        # Set generic simulation parameters (mass, gravity constant)
        m = generic_parameter_symbols["m"]
        g = generic_parameter_symbols["g"]

        # define interaction-specific constants (spring constant, damping factor
        # and stable spring position)
        parameter_labels = ["k_t", "d_t", "l_stable"]
        parameter_labels = [label + str(node_id) for label in parameter_labels]
        k_t, d_t, l_stable = sp.symbols(
            parameter_labels[0] + "," + parameter_labels[1] + "," + parameter_labels[2]
        )
        self.parameter_symbols = {
            parameter_labels[0]: k_t,
            parameter_labels[1]: d_t,
            parameter_labels[2]: l_stable
        }
        self.parameter_values = {
            parameter_labels[0]: 1.,
            parameter_labels[1]: 0.01,
            parameter_labels[2]: 0.5
        }

        # define dynamical variables. r: position, v: velocity (time-derivative of r)
        self.r = me.dynamicsymbols("r" + str(node_id))
        self.v = me.dynamicsymbols("v" + str(node_id))

        # Sample a given parent interframe using a uniform distribution on extrinsic
        # euler coordinates. Parent interfame always has a fixed transformation from
        # the parent's frame (valid at all simulation timesteps).
        if sample_interframe:
            rot_amounts = tuple(np.random.uniform(0, 2 * np.pi, 3))
        else:
            rot_amounts = (0, 0, 0)
        self.parent_interframe_rot = Rot.from_euler("XYZ", rot_amounts)
        parent_interfame = parent.frame.orientnew(
            "I" + str(node_id), "Body", rot_amounts, rot_order=123
        )

        # Define joint using pre-defined sympy PrismaticJoint class
        self.joint = me.PrismaticJoint(
            "J" + str(node_id),
            parent,
            child,
            coordinates=self.r,
            speeds=self.v,
            parent_interframe=parent_interfame,
            joint_axis=parent_interfame.x,
        )

        # Define force equations (always use springs otherwise may drift)
        spring_eq = - k_t * (self.r - l_stable) * parent_interfame.x
        damper_eq = - d_t * self.v * parent_interfame.x if use_damper else 0
        gravity_eq = - m * g * world.z if use_gravity else 0
        self.force_eq = spring_eq + damper_eq + gravity_eq

        if self.force_eq:
            self.child.apply_force(self.force_eq, reaction_body=self.parent)

        # Apply external forces to child body
        if apply_force:
            force_labels = ["f_x", "f_y", "f_z"]
            force_labels = [f + str(node_id) for f in force_labels]
            f_x, f_y, f_z = sp.symbols(
                force_labels[0] + "," + force_labels[1] + "," + force_labels[2]
            )
            self.force_symbols = {
                force_labels[0]: f_x,
                force_labels[1]: f_y,
                force_labels[2]: f_z,
            }
            self.force_values = {
                force_labels[0]: 0,
                force_labels[1]: 0,
                force_labels[2]: 0,
            }

            self.child.apply_force(
                f_x * world.x + f_y * world.y + f_z * world.z
            )

            self.parameter_symbols.update(self.force_symbols)
            self.parameter_values.update(self.force_values)

    def get_interaction_parameters(self):
        """Retrieve interaction-specific simulation parameters."""
        return self.parameter_symbols, self.parameter_values

    def get_dynamic_variables(self):
        """Retrieve dynamical variables. There may be more than one dynamic variable
        for specific types of joints (e.g. spherical joints) so always expressed them
        in list format."""
        return [self.r], [self.v]

    def get_joint(self):
        return self.joint

    def get_initial_conditions(self):
        return self.init_cond

    def get_child_pose(self, parent_state, child_r, child_v, link_length):
        """Get the current state of the child body based on parent's current state
        and current numerical values of the dynamical variables of the joint.
        """
        # Get parent location and frame vectors (expressed in world coordinates)
        parent_pos = parent_state.location
        parent_frame = parent_state.morphological_features["pose_vectors"]

        # Apply euler rotation to get the child frame vectors (in parent's frame)
        # For a prismatic joint, child frame is the same frame as parent interframe
        parent_frame_rot = Rot.from_matrix(parent_frame)
        child_frame_rot = parent_frame_rot * self.parent_interframe_rot
        child_frame = child_frame_rot.as_matrix()

        # Compute 3D position of child object based on parent's current state
        child_pos = parent_pos + child_r * child_frame[:, 0]

        # child's linear velocity
        parent_lin_vel = parent_state.non_morphological_features["linear_velocity"]
        child_lin_vel = parent_lin_vel + child_v * child_frame[:, 0]

        # child's angular velocity vector
        omega_vec = np.zeros(3)

        return child_pos, child_frame, child_lin_vel, omega_vec


class PlanarJoint():
    def __init__(
        self,
        node_id,
        parent,
        child,
        world,
        generic_parameter_symbols,
        use_spring=False,
        use_damper=False,
        use_gravity=True,
        apply_force=True,
        sample_interframe=True,
        r_init=0,
        v_init=1e-3
    ):
        """Class implementing planar rotational joint connecting two object parts
        (parent and child). It builds on the PinJoint class, as defined here:
        https://docs.sympy.org/latest/modules/physics/mechanics/api/joint.html

        Only has single degree of freedom in an arbitrary angular rotation axis. This
        direction is arbitrary and is defined as the y axis of the parent's body
        interframe (itself randomly sampled). While the joint itself only constraints
        the dynamics of the two connected bodies to specific areas of the state space,
        additional force elements can be added to enrich the dynamics.
        This is the interaction class to use when defining planar pendulum joints.

        :param node_id: index of child in the chain (only used to name sympy variables).
        :param parent: parent object part, body instance. Part which the edge originates
        from.
        :param child: child object part, body instance. Part which the edge directs to.
        :param world: global reference frame of the simulation
        :param generic_parameter_symbols: List of generic simulation parameters. For
        now, these are limited to the link_length, mass per node and gravity constant.
        Need to be provided as they influence the forces applied on the child object.
        :param use_spring: boolean, defines whether to add a linear spring to the joint.
        :param use_damper: boolean, defines whether to use a damping element on top of
        the joint.
        :param use_gravity: whether to add gravity to the forces acting on child object.
        :param apply_force: whether to allow applying force on child object.
        :param sample_interframe: whether to sample an interframe (i.e. arbitrary
        rotation) for the current joint.
        :param r_init: initial conditions to use for the position/angle (dynamical
        variable) of the angular joint.
        :param v_init: initial conditions to use for the velocity (time-derivative of
        angle dynamical variable) of the joint.
        """
        self.node_id = node_id
        self.parent = parent
        self.child = child
        self.world = world
        self.use_spring = use_spring
        self.use_damper = use_damper
        self.init_cond = [r_init, v_init]

        # Set generic simulation parameters (link_length, gravity)
        link_length = generic_parameter_symbols["l"]
        gravity = generic_parameter_symbols["g"]

        # Define interaction-specific parameters
        parameter_labels = ["k_r", "d_r", "r_stable"]
        parameter_labels = [label + str(node_id) for label in parameter_labels]
        k_r, d_r, r_eq = sp.symbols(
            parameter_labels[0] + "," + parameter_labels[1] + "," + parameter_labels[2]
        )
        self.parameter_symbols = {
            parameter_labels[0]: k_r,
            parameter_labels[1]: d_r,
            parameter_labels[2]: r_eq
        }
        self.parameter_values = {
            parameter_labels[0]: 1.,
            parameter_labels[1]: 0.005,
            parameter_labels[2]: -np.pi / 4
        }

        # define dynamic variables
        self.r = me.dynamicsymbols("r" + str(node_id))
        self.v = me.dynamicsymbols("v" + str(node_id))

        # Sample a given parent interframe based on a uniform distribution of extrinsic
        # euler coordinates
        if sample_interframe:
            rot_amounts = tuple(np.random.uniform(0, 2 * np.pi, 3))
        else:
            rot_amounts = (0, 0, 0)

        self.parent_interframe_rot = Rot.from_euler("XYZ", rot_amounts)
        parent_interframe = parent.frame.orientnew(
            "I" + str(node_id), "Body", rot_amounts, rot_order=123
        )

        # Define the interaction itself using pre-defined sympy Joint class
        self.joint = me.PinJoint(
            "J" + str(node_id), parent, child, coordinates=self.r, speeds=self.v,
            child_point=-link_length * child.x, parent_interframe=parent_interframe,
            joint_axis=parent_interframe.y
        )

        # Define force and torque equations
        spring_eq = - k_r * (self.r - r_eq) * parent_interframe.y if use_spring else 0
        damper_eq = - d_r * self.v * parent_interframe.y if use_damper else 0
        torque_eq = spring_eq + damper_eq
        if torque_eq:
            self.child.apply_torque(torque_eq, reaction_body=self.parent)

        if use_gravity:
            self.child.apply_force(- child.mass * gravity * world.z)

        # External forces to defining action space
        if apply_force:
            force_labels = ["f_x", "f_y", "f_z"]
            force_labels = [f + str(node_id) for f in force_labels]
            f_x, f_y, f_z = sp.symbols(
                force_labels[0] + "," + force_labels[1] + "," + force_labels[2]
            )
            self.force_symbols = {
                force_labels[0]: f_x,
                force_labels[1]: f_y,
                force_labels[2]: f_z,
            }
            self.force_values = {
                force_labels[0]: 0,
                force_labels[1]: 0,
                force_labels[2]: 0,
            }

            self.parameter_symbols.update(self.force_symbols)
            self.parameter_values.update(self.force_values)

            self.child.apply_force(
                f_x * world.x
                + f_y * world.y
                + f_z * world.z
            )

    def get_interaction_parameters(self):
        """Retrieve interaction-specific simulation parameters."""
        return self.parameter_symbols, self.parameter_values

    def get_dynamic_variables(self):
        """Retrieve dynamical variables. There may be more than one dynamic variable
        for specific types of joints (e.g. spherical joints) so always expressed them
        in list format."""
        return [self.r], [self.v]

    def get_joint(self):
        return self.joint

    def get_initial_conditions(self):
        return self.init_cond

    def get_child_pose(self, parent_state, child_r, child_v, link_length):
        """Get the current state of the child body based on parent's current state
        and current numerical values of the dynamical variables of the joint.
        """
        # Get parent frame vectors (expressed in world coordinates)
        parent_frame = parent_state.morphological_features["pose_vectors"]
        parent_frame_rot = Rot.from_matrix(parent_frame)

        # Define joint_rotation
        joint_rot = Rot.from_euler("y", child_r)

        # Define parent_interframe rotation (vs parent frame NOT world frame)
        parent_interframe_rot = self.parent_interframe_rot * joint_rot

        # Get child_frame by applying interframe rotation to parent_frame
        child_frame_rot = parent_frame_rot * parent_interframe_rot
        child_frame = child_frame_rot.as_matrix()

        # Compute position of child object based on previous one
        parent_pos = parent_state.location
        child_pos = parent_pos + link_length * child_frame[:, 0]

        # get both linear and angular velocities
        parent_lin_vel = parent_state.non_morphological_features["linear_velocity"]
        omega_vec = child_v * child_frame[:, 1]
        delta_r = parent_pos - child_pos
        child_lin_vel = parent_lin_vel + np.cross(omega_vec, delta_r)

        return child_pos, child_frame, child_lin_vel, omega_vec

# TODO: create a parent Joint() class, which all interactions would inherit from.
# Would contain the generic functionalities.
# TODO: create classes for spherical joints and rigid/weld joints. Can use
# me.SphericalJoint and me.WeldJoint classes.
