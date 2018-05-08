from __future__ import print_function, division

import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

from numpy import array, linspace, deg2rad, zeros
from sympy import symbols
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, RigidBody, KanesMethod
from scipy.integrate import odeint
from pydy.codegen.ode_function_generators import generate_ode_function

import matplotlib.pyplot as plt


class MultipendulumEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):

        self.num_links = 5  # Number of links
        total_link_length = 1.
        total_link_mass = 1.
        self.ind_link_length = total_link_length / self.num_links
        ind_link_com_length = self.ind_link_length / 2.
        ind_link_mass = total_link_mass / self.num_links
        ind_link_inertia = ind_link_mass * (ind_link_com_length ** 2)

        # =======================#
        # Parameters for step() #
        # =======================#

        # Maximum number of steps before episode termination
        self.max_steps = 200

        # For ODE integration
        self.dt = .0001  # Simultaion time step = 1ms
        self.sim_steps = 51  # Number of simulation steps in 1 learning step
        self.dt_step = np.linspace(0., self.dt * self.sim_steps, num=self.sim_steps)  # Learning time step = 50ms

        # Termination conditions for simulation
        self.num_steps = 0  # Step counter
        self.done = False

        # For visualisation
        self.viewer = None
        self.ax = False

        # Constraints for observation
        min_angle = -np.pi  # Angle
        max_angle = np.pi
        min_omega = -10.  # Angular velocity
        max_omega = 10.
        min_torque = -10.  # Torque
        max_torque = 10.

        low_state_angle = np.full(self.num_links, min_angle)  # Min angle
        low_state_omega = np.full(self.num_links, min_omega)  # Min angular velocity
        low_state = np.append(low_state_angle, low_state_omega)
        high_state_angle = np.full(self.num_links, max_angle)  # Max angle
        high_state_omega = np.full(self.num_links, max_omega)  # Max angular velocity
        high_state = np.append(high_state_angle, high_state_omega)
        low_action = np.full(self.num_links, min_torque)  # Min torque
        high_action = np.full(self.num_links, max_torque)  # Max torque
        self.action_space = spaces.Box(low=low_action, high=high_action)
        self.observation_space = spaces.Box(low=low_state, high=high_state)

        # Minimum reward
        self.min_reward = -(max_angle ** 2 + .1 * max_omega ** 2 + .001 * max_torque ** 2) * self.num_links

        # Seeding
        self.seed()

        # ==============#
        # Orientations #
        # ==============#
        self.inertial_frame = ReferenceFrame('I')
        self.link_frame = []
        self.theta = []
        for i in range(self.num_links):
            temp_angle_name = "theta{}".format(i + 1)
            temp_link_name = "L{}".format(i + 1)
            self.theta.append(dynamicsymbols(temp_angle_name))
            self.link_frame.append(ReferenceFrame(temp_link_name))
            if i == 0:  # First link
                self.link_frame[i].orient(self.inertial_frame, 'Axis', (self.theta[i], self.inertial_frame.z))
            else:  # Second link, third link...
                self.link_frame[i].orient(self.link_frame[i - 1], 'Axis', (self.theta[i], self.link_frame[i - 1].z))

        # =================#
        # Point Locations #
        # =================#

        # --------#
        # Joints #
        # --------#
        self.link_length = []
        self.link_joint = []
        for i in range(self.num_links):
            temp_link_length_name = "l_L{}".format(i + 1)
            temp_link_joint_name = "A{}".format(i)
            self.link_length.append(symbols(temp_link_length_name))
            self.link_joint.append(Point(temp_link_joint_name))
            if i > 0:  # Set position started from link2, then link3, link4...
                self.link_joint[i].set_pos(self.link_joint[i - 1], self.link_length[i - 1] * self.link_frame[i - 1].y)

        # --------------------------#
        # Centre of mass locations #
        # --------------------------#
        self.link_com_length = []
        self.link_mass_centre = []
        for i in range(self.num_links):
            temp_link_com_length_name = "d_L{}".format(i + 1)
            temp_link_mass_centre_name = "L{}_o".format(i + 1)
            self.link_com_length.append(symbols(temp_link_com_length_name))
            self.link_mass_centre.append(Point(temp_link_mass_centre_name))
            self.link_mass_centre[i].set_pos(self.link_joint[i], self.link_com_length[i] * self.link_frame[i].y)

        # ===========================================#
        # Define kinematical differential equations #
        # ===========================================#
        self.omega = []
        self.kinematical_differential_equations = []
        self.time = symbols('t')
        for i in range(self.num_links):
            temp_omega_name = "omega{}".format(i + 1)
            self.omega.append(dynamicsymbols(temp_omega_name))
            self.kinematical_differential_equations.append(self.omega[i] - self.theta[i].diff(self.time))

        # ====================#
        # Angular Velocities #
        # ====================#
        for i in range(self.num_links):
            if i == 0:  # First link
                self.link_frame[i].set_ang_vel(self.inertial_frame, self.omega[i] * self.inertial_frame.z)
            else:  # Second link, third link...
                self.link_frame[i].set_ang_vel(self.link_frame[i - 1], self.omega[i] * self.link_frame[i - 1].z)

        # ===================#
        # Linear Velocities #
        # ===================#
        for i in range(self.num_links):
            if i == 0:  # First link
                self.link_joint[i].set_vel(self.inertial_frame, 0)
            else:  # Second link, third link...
                self.link_joint[i].v2pt_theory(self.link_joint[i - 1], self.inertial_frame, self.link_frame[i - 1])
            self.link_mass_centre[i].v2pt_theory(self.link_joint[i], self.inertial_frame, self.link_frame[i])

        # ======#
        # Mass #
        # ======#
        self.link_mass = []
        for i in range(self.num_links):
            temp_link_mass_name = "m_L{}".format(i + 1)
            self.link_mass.append(symbols(temp_link_mass_name))

        # =========#
        # Inertia #
        # =========#
        self.link_inertia = []
        self.link_inertia_dyadic = []
        self.link_central_inertia = []
        for i in range(self.num_links):
            temp_link_inertia_name = "I_L{}z".format(i + 1)
            self.link_inertia.append(symbols(temp_link_inertia_name))
            self.link_inertia_dyadic.append(inertia(self.link_frame[i], 0, 0, self.link_inertia[i]))
            self.link_central_inertia.append((self.link_inertia_dyadic[i], self.link_mass_centre[i]))

        # ==============#
        # Rigid Bodies #
        # ==============#
        self.link = []
        for i in range(self.num_links):
            temp_link_name = "link{}".format(i + 1)
            self.link.append(RigidBody(temp_link_name, self.link_mass_centre[i], self.link_frame[i],
                                       self.link_mass[i], self.link_central_inertia[i]))

        # =========#
        # Gravity #
        # =========#
        self.g = symbols('g')
        self.link_grav_force = []
        for i in range(self.num_links):
            self.link_grav_force.append((self.link_mass_centre[i],
                                         -self.link_mass[i] * self.g * self.inertial_frame.y))

        # ===============#
        # Joint Torques #
        # ===============#
        self.link_joint_torque = []
        self.link_torque = []
        for i in range(self.num_links):
            temp_link_joint_torque_name = "T_a{}".format(i + 1)
            self.link_joint_torque.append(dynamicsymbols(temp_link_joint_torque_name))
        for i in range(self.num_links):
            if (i + 1) == self.num_links:  # Last link
                self.link_torque.append((self.link_frame[i],
                                         self.link_joint_torque[i] * self.inertial_frame.z))
            else:  # Other links
                self.link_torque.append((self.link_frame[i],
                                         self.link_joint_torque[i] * self.inertial_frame.z
                                         - self.link_joint_torque[i + 1] * self.inertial_frame.z))

        # =====================#
        # Equations of Motion #
        # =====================#
        self.coordinates = []
        self.speeds = []
        self.loads = []
        self.bodies = []
        for i in range(self.num_links):
            self.coordinates.append(self.theta[i])
            self.speeds.append(self.omega[i])
            self.loads.append(self.link_grav_force[i])
            self.loads.append(self.link_torque[i])
            self.bodies.append(self.link[i])
        self.kane = KanesMethod(self.inertial_frame,
                                self.coordinates,
                                self.speeds,
                                self.kinematical_differential_equations)
        self.fr, self.frstar = self.kane.kanes_equations(self.bodies, self.loads)
        self.mass_matrix = self.kane.mass_matrix_full
        self.forcing_vector = self.kane.forcing_full

        # =============================#
        # List the symbolic arguments #
        # =============================#
        # -----------#
        # Constants #
        # -----------#
        self.constants = []
        for i in range(self.num_links):
            if (i + 1) != self.num_links:
                self.constants.append(self.link_length[i])
            self.constants.append(self.link_com_length[i])
            self.constants.append(self.link_mass[i])
            self.constants.append(self.link_inertia[i])
        self.constants.append(self.g)

        # --------------#
        # Time Varying #
        # --------------#
        self.coordinates = []
        self.speeds = []
        self.specified = []
        for i in range(self.num_links):
            self.coordinates.append(self.theta[i])
            self.speeds.append(self.omega[i])
            self.specified.append(self.link_joint_torque[i])

        # =======================#
        # Generate RHS Function #
        # =======================#
        self.right_hand_side = generate_ode_function(self.forcing_vector, self.coordinates, self.speeds,
                                                     self.constants, mass_matrix=self.mass_matrix,
                                                     specifieds=self.specified)

        # ==============================#
        # Specify Numerical Quantities #
        # ==============================#
        self.x = np.zeros(self.num_links * 2)
        self.x[:self.num_links] = deg2rad(2.0)

        self.numerical_constants = []
        for i in range(self.num_links):
            if (i + 1) != self.num_links:
                self.numerical_constants.append(self.ind_link_length)
            self.numerical_constants.append(ind_link_com_length)
            self.numerical_constants.append(ind_link_mass)
            self.numerical_constants.append(ind_link_inertia)
        self.numerical_constants.append(9.81)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.num_steps = 0  # Step counter
        self.done = False  # Done flag
        self.x = np.random.randn(self.num_links * 2)  # State
        return self._get_obs()

    def _get_obs(self):
        return self.x

    def sample_action(self):
        return np.random.randn(self.num_links)

    def step(self, action):
        if self.done == True or self.num_steps > self.max_steps:
            self.done = True
            # Normalised reward
            reward = 0.
            return self.x, reward, self.done, {}
        else:
            # Increment the step counter
            self.num_steps += 1
            # Simulation
            self.x = odeint(self.right_hand_side, self.x, self.dt_step, args=(action, self.numerical_constants))[-1]
            # Normalise joint angles to -pi ~ pi
            self.x[:self.num_links] = self.angle_normalise(self.x[:self.num_links])

            # n-link case
            reward = 0.
            # Cost due to angle and torque
            for i in range(self.num_links):
                reward -= (self.x[i] ** 2 + .001 * action[i] ** 2)
            # Cost due to angular velocity
            for i in range(self.num_links, self.num_links * 2):
                reward -= (.1 * self.x[i] ** 2)
            # Normalised reward
            reward = (reward - self.min_reward) / (-self.min_reward)

            return self.x, reward, self.done, {}

    def angle_normalise(self, angle_input):
        return (((angle_input + np.pi) % (2 * np.pi)) - np.pi)

    def render(self, mode='human'):
        if not self.ax:
            fig, ax = plt.subplots()
            ax.set_xlim([-1.5, 1.5])
            ax.set_ylim([-1.5, 1.5])
            ax.set_aspect('equal')
            self.ax = ax
        else:
            self.ax.clear()
            self.ax.set_xlim([-1.5, 1.5])
            self.ax.set_ylim([-1.5, 1.5])
            self.ax.set_aspect('equal')

        x = [0.]
        y = [0.]
        for i in range(self.num_links):
            x.append(x[i] + self.ind_link_length*np.cos(self.x[i] + np.pi / 2.))
            y.append(y[i] + self.ind_link_length*np.sin(self.x[i] + np.pi / 2.))
        plt.plot(x, y)

        plt.pause(0.01)
