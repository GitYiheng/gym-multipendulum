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
        #=======================#
        # Parameters for step() #
        #=======================#
        # Maximum number of steps before episode termination
        self.max_steps = 200
        # For ODE integration
        self.dt = .001 # Simultaion time step = 1ms
        self.sim_steps = 51 # Number of simulation steps in 1 learning step
        self.dt_step = np.linspace(0., self.dt*self.sim_steps, num=self.sim_steps) # Learning time step = 50ms
        # Termination conditions for simulation
        self.num_steps = 0 # Number of steps
        self.done = False
        # For visualisation
        self.viewer = None
        self.ax = False
        # Constraints for observation
        min_angle = -np.pi
        max_angle = np.pi
        min_omega = -10.
        max_omega = 10.
        min_torque = -10.
        max_torque = 10.
        low_state = np.array([min_angle, min_angle, min_angle, min_omega, min_omega, min_omega])
        high_state = np.array([max_angle, max_angle, max_angle, max_omega, max_omega, max_omega])
        low_action = np.array([min_torque, min_torque, min_torque])
        high_action = np.array([max_torque, max_torque, max_torque])
        self.action_space = spaces.Box(low=low_action, high=high_action)
        self.observation_space = spaces.Box(low=low_state, high=high_state)
        # Seed...
        self.seed()
        #==============#
        # Orientations #
        #==============#
        self.theta1, self.theta2, self.theta3 = dynamicsymbols('theta1, theta2, theta3')
        self.inertial_frame = ReferenceFrame('I')
        self.link1_frame = ReferenceFrame('L')
        self.link1_frame.orient(self.inertial_frame, 'Axis', (self.theta1, self.inertial_frame.z))
        self.link2_frame = ReferenceFrame('U')
        self.link2_frame.orient(self.link1_frame, 'Axis', (self.theta2, self.link1_frame.z))
        self.link3_frame = ReferenceFrame('T')
        self.link3_frame.orient(self.link2_frame, 'Axis', (self.theta3, self.link2_frame.z))
        #=================#
        # Point Locations #
        #=================#
        #--------#
        # Joints #
        #--------#
        self.link1_length, self.link2_length = symbols('l_L, l_U')
        self.link1_joint = Point('A')
        self.link2_joint = Point('K')
        self.link2_joint.set_pos(self.link1_joint, self.link1_length * self.link1_frame.y)
        self.link3_joint = Point('H')
        self.link3_joint.set_pos(self.link2_joint, self.link2_length * self.link2_frame.y)
        #--------------------------#
        # Centre of mass locations #
        #--------------------------#
        self.link1_com_length, self.link2_com_length, self.link3_com_length = symbols('d_L, d_U, d_T')
        self.link1_mass_centre = Point('L_o')
        self.link1_mass_centre.set_pos(self.link1_joint, self.link1_com_length * self.link1_frame.y)
        self.link2_mass_centre = Point('U_o')
        self.link2_mass_centre.set_pos(self.link2_joint, self.link2_com_length * self.link2_frame.y)
        self.link3_mass_centre = Point('T_o')
        self.link3_mass_centre.set_pos(self.link3_joint, self.link3_com_length * self.link3_frame.y)
        #===========================================#
        # Define kinematical differential equations #
        #===========================================#
        self.omega1, self.omega2, self.omega3 = dynamicsymbols('omega1, omega2, omega3')
        self.time = symbols('t')
        self.kinematical_differential_equations = [self.omega1 - self.theta1.diff(self.time),
            self.omega2 - self.theta2.diff(self.time),
            self.omega3 - self.theta3.diff(self.time)]
        #====================#
        # Angular Velocities #
        #====================#
        self.link1_frame.set_ang_vel(self.inertial_frame, self.omega1 * self.inertial_frame.z)
        self.link2_frame.set_ang_vel(self.link1_frame, self.omega2 * self.link1_frame.z)
        self.link3_frame.set_ang_vel(self.link2_frame, self.omega3 * self.link2_frame.z)
        #===================#
        # Linear Velocities #
        #===================#
        self.link1_joint.set_vel(self.inertial_frame, 0)
        self.link1_mass_centre.v2pt_theory(self.link1_joint, self.inertial_frame, self.link1_frame)
        self.link2_joint.v2pt_theory(self.link1_joint, self.inertial_frame, self.link1_frame)
        self.link2_mass_centre.v2pt_theory(self.link2_joint, self.inertial_frame, self.link2_frame)
        self.link3_joint.v2pt_theory(self.link2_joint, self.inertial_frame, self.link2_frame)
        self.link3_mass_centre.v2pt_theory(self.link3_joint, self.inertial_frame, self.link3_frame)
        #======#
        # Mass #
        #======#
        self.link1_mass, self.link2_mass, self.link3_mass = symbols('m_L, m_U, m_T')
        #=========#
        # Inertia #
        #=========#
        self.link1_inertia, self.link2_inertia, self.link3_inertia = symbols('I_Lz, I_Uz, I_Tz')
        self.link1_inertia_dyadic = inertia(self.link1_frame, 0, 0, self.link1_inertia)
        self.link1_central_inertia = (self.link1_inertia_dyadic, self.link1_mass_centre)
        self.link2_inertia_dyadic = inertia(self.link2_frame, 0, 0, self.link2_inertia)
        self.link2_central_inertia = (self.link2_inertia_dyadic, self.link2_mass_centre)
        self.link3_inertia_dyadic = inertia(self.link3_frame, 0, 0, self.link3_inertia)
        self.link3_central_inertia = (self.link3_inertia_dyadic, self.link3_mass_centre)
        #==============#
        # Rigid Bodies #
        #==============#
        self.link1 = RigidBody('link1', self.link1_mass_centre, self.link1_frame,
            self.link1_mass, self.link1_central_inertia)
        self.link2 = RigidBody('link2', self.link2_mass_centre, self.link2_frame,
            self.link2_mass, self.link2_central_inertia)
        self.link3 = RigidBody('link3', self.link3_mass_centre, self.link3_frame,
            self.link3_mass, self.link3_central_inertia)
        #=========#
        # Gravity #
        #=========#
        self.g = symbols('g')
        self.link1_grav_force = (self.link1_mass_centre,
            -self.link1_mass * self.g * self.inertial_frame.y)
        self.link2_grav_force = (self.link2_mass_centre,
            -self.link2_mass * self.g * self.inertial_frame.y)
        self.link3_grav_force = (self.link3_mass_centre, -self.link3_mass * self.g * self.inertial_frame.y)
        #===============#
        # Joint Torques #
        #===============#
        self.link1_joint_torque, self.link2_joint_torque, self.link3_joint_torque = dynamicsymbols('T_a, T_k, T_h')
        self.link1_torque = (self.link1_frame,
            self.link1_joint_torque * self.inertial_frame.z - self.link2_joint_torque *
            self.inertial_frame.z)
        self.link2_torque = (self.link2_frame,
            self.link2_joint_torque * self.inertial_frame.z - self.link3_joint_torque *
            self.inertial_frame.z)
        self.link3_torque = (self.link3_frame, self.link3_joint_torque * self.inertial_frame.z)
        #=====================#
        # Equations of Motion #
        #=====================#
        self.coordinates = [self.theta1, self.theta2, self.theta3]
        self.speeds = [self.omega1, self.omega2, self.omega3]
        self.kane = KanesMethod(self.inertial_frame,
            self.coordinates,
            self.speeds,
            self.kinematical_differential_equations)
        self.loads = [self.link1_grav_force,
            self.link2_grav_force,
            self.link3_grav_force,
            self.link1_torque,
            self.link2_torque,
            self.link3_torque]
        self.bodies = [self.link1, self.link2, self.link3]
        self.fr, self.frstar = self.kane.kanes_equations(self.bodies, self.loads)
        self.mass_matrix = self.kane.mass_matrix_full
        self.forcing_vector = self.kane.forcing_full
        #=============================#
        # List the symbolic arguments #
        #=============================#
        #-----------#
        # Constants #
        #-----------#
        self.constants = [self.link1_length,
            self.link1_com_length,
            self.link1_mass,
            self.link1_inertia,
            self.link2_length,
            self.link2_com_length,
            self.link2_mass,
            self.link2_inertia,
            self.link3_com_length,
            self.link3_mass,
            self.link3_inertia,
            self.g]
        #--------------#
        # Time Varying #
        #--------------#
        self.coordinates = [self.theta1, self.theta2, self.theta3]
        self.speeds = [self.omega1, self.omega2, self.omega3]
        self.specified = [self.link1_joint_torque, self.link2_joint_torque, self.link3_joint_torque]
        #=======================#
        # Generate RHS Function #
        #=======================#
        self.right_hand_side = generate_ode_function(self.forcing_vector, self.coordinates, self.speeds,
            self.constants, mass_matrix=self.mass_matrix,
            specifieds=self.specified)
        #==============================#
        # Specify Numerical Quantities #
        #==============================#
        self.x = zeros(6)
        self.x[:3] = deg2rad(2.0)
        # taken from male1.txt in yeadon (maybe I should use the values in Winters).
        # self.numerical_constants = array([0.611,  # link1_length [m]
        #     0.387,  # link1_com_length [m]
        #     6.769,  # link1_mass [kg]
        #     0.101,  # link1_inertia [kg*m^2]
        #     0.424,  # link2_length [m]
        #     0.193,  # link2_com_length
        #     17.01,  # link2_mass [kg]
        #     0.282,  # link2_inertia [kg*m^2]
        #     0.305,  # link3_com_length [m]
        #     32.44,  # link3_mass [kg]
        #     1.485,  # link3_inertia [kg*m^2]
        #     9.81])  # acceleration due to gravity [m/s^2]
        self.numerical_constants = array([0.500,  # link1_length [m]
            0.250,  # link1_com_length [m]
            0.500,  # link1_mass [kg]
            0.03125,  # link1_inertia [kg*m^2]
            0.500,  # link2_length [m]
            0.250,  # link2_com_length
            0.500,  # link2_mass [kg]
            0.03125,  # link2_inertia [kg*m^2]
            0.250,  # link3_com_length [m]
            0.500,  # link3_mass [kg]
            0.03125,  # link3_inertia [kg*m^2]
            9.81])  # acceleration due to gravity [m/s^2]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # self.x = zeros(6)
        # self.x[:3] = deg2rad(2.0)
        self.num_steps = 0
        self.done = False
        self.x = np.random.randn(6)
        self.x[:3] += np.array([np.pi, np.pi, np.pi])
        return self._get_obs()

    def _get_obs(self):
        return self.x

    def sample_action(self):
        return np.random.randn(3)

    def step(self, action):
        if self.done == True or self.num_steps > self.max_steps:
            self.done = True
            # Normalised reward
            reward = 0.
            # Unnormalised reward
            # reward = -60.
            return self.x, reward, self.done, {}
        else:
            # Increment the step counter
            self.num_steps += 1
            # Simulation
            self.x = odeint(self.right_hand_side, self.x, self.dt_step,
                args=(action, self.numerical_constants))[-1]
            # Normalise joint angles to -pi ~ pi
            self.x[:3] = self.angle_normalise(self.x[:3])
            # Normalise the reward to 0. ~ 1.
            # Max reward: 0. -> 1.
            # Min reward: -59.90881320326807 -> 0.
            reward_unnormed = 60. - (self.x[0] ** 2 + self.x[1] ** 2 + self.x[2] ** 2 + .1 * self.x[3] ** 2 + .1 * self.x[4] ** 2 + .1 * self.x[5] ** 2 + .001 * action[0] ** 2 + .001 * action[1] ** 2 + .001 * action[2] ** 2)
            reward = reward_unnormed / 60.
            # Unnormalized reward
            # reward = - (self.x[0] ** 2 + self.x[1] ** 2 + self.x[2] ** 2 + .1 * self.x[3] ** 2 + .1 * self.x[4] ** 2 + .1 * self.x[5] ** 2 + .001 * action[0] ** 2 + .001 * action[1] ** 2 + .001 * action[2] ** 2)
        return self.x, reward, self.done, {}

    def angle_normalise(self, angle_input):
        return (((angle_input+np.pi) % (2*np.pi)) - np.pi)

    def render(self, mode='human'):
        if not self.ax:
            fig, ax = plt.subplots()
            ax.set_xlim([-3.5, 3.5])
            ax.set_ylim([-3.5, 3.5])
            ax.set_aspect('equal')
            self.ax = ax
        else:
            self.ax.clear()
            self.ax.set_xlim([-3.5, 3.5])
            self.ax.set_ylim([-3.5, 3.5])
            self.ax.set_aspect('equal')
        x0 = 0.
        y0 = 0.
        x1 = x0 + np.cos(self.x[0]+np.pi/2.)
        y1 = y0 + np.sin(self.x[0]+np.pi/2.)
        x2 = x1 + np.cos(self.x[1]+np.pi/2.)
        y2 = y1 + np.sin(self.x[1]+np.pi/2.)
        x3 = x2 + np.cos(self.x[2]+np.pi/2.)
        y3 = y2 + np.sin(self.x[2]+np.pi/2.)
        plt.plot([x0, x1, x2, x3], [y0, y1, y2, y3])
        plt.pause(0.01)
