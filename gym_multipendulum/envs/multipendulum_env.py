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
        self.dt = .05
        self.viewer = None
        self.ax = False

        min_angle = -5
        max_angle = 5
        min_omega = -10
        max_omega = 10
        min_torque = -10
        max_torque = 10
        low_state = np.array([min_angle, min_angle, min_angle, min_omega, min_omega, min_omega])
        high_state = np.array([max_angle, max_angle, max_angle, max_omega, max_omega, max_omega])
        low_action = np.array([min_torque, min_torque, min_torque])
        high_action = np.array([max_torque, max_torque, max_torque])
        self.action_space = spaces.Box(low=low_action, high=high_action)
        self.observation_space = spaces.Box(low=low_state, high=high_state)
        self.seed()
        #==============#
        # Orientations #
        #==============#
        self.theta1, self.theta2, self.theta3 = dynamicsymbols('theta1, theta2, theta3')
        self.inertial_frame = ReferenceFrame('I')
        self.lower_leg_frame = ReferenceFrame('L')
        self.lower_leg_frame.orient(self.inertial_frame, 'Axis', (self.theta1, self.inertial_frame.z))
        self.upper_leg_frame = ReferenceFrame('U')
        self.upper_leg_frame.orient(self.lower_leg_frame, 'Axis', (self.theta2, self.lower_leg_frame.z))
        self.torso_frame = ReferenceFrame('T')
        self.torso_frame.orient(self.upper_leg_frame, 'Axis', (self.theta3, self.upper_leg_frame.z))
        #=================#
        # Point Locations #
        #=================#
        #--------#
        # Joints #
        #--------#
        self.lower_leg_length, self.upper_leg_length = symbols('l_L, l_U')
        self.ankle = Point('A')
        self.knee = Point('K')
        self.knee.set_pos(self.ankle, self.lower_leg_length * self.lower_leg_frame.y)
        self.hip = Point('H')
        self.hip.set_pos(self.knee, self.upper_leg_length * self.upper_leg_frame.y)
        #--------------------------#
        # Center of mass locations #
        #--------------------------#
        self.lower_leg_com_length, self.upper_leg_com_length, self.torso_com_length = symbols('d_L, d_U, d_T')
        self.lower_leg_mass_center = Point('L_o')
        self.lower_leg_mass_center.set_pos(self.ankle, self.lower_leg_com_length * self.lower_leg_frame.y)
        self.upper_leg_mass_center = Point('U_o')
        self.upper_leg_mass_center.set_pos(self.knee, self.upper_leg_com_length * self.upper_leg_frame.y)
        self.torso_mass_center = Point('T_o')
        self.torso_mass_center.set_pos(self.hip, self.torso_com_length * self.torso_frame.y)
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
        self.lower_leg_frame.set_ang_vel(self.inertial_frame, self.omega1 * self.inertial_frame.z)
        self.upper_leg_frame.set_ang_vel(self.lower_leg_frame, self.omega2 * self.lower_leg_frame.z)
        self.torso_frame.set_ang_vel(self.upper_leg_frame, self.omega3 * self.upper_leg_frame.z)
        #===================#
        # Linear Velocities #
        #===================#
        self.ankle.set_vel(self.inertial_frame, 0)
        self.lower_leg_mass_center.v2pt_theory(self.ankle, self.inertial_frame, self.lower_leg_frame)
        self.knee.v2pt_theory(self.ankle, self.inertial_frame, self.lower_leg_frame)
        self.upper_leg_mass_center.v2pt_theory(self.knee, self.inertial_frame, self.upper_leg_frame)
        self.hip.v2pt_theory(self.knee, self.inertial_frame, self.upper_leg_frame)
        self.torso_mass_center.v2pt_theory(self.hip, self.inertial_frame, self.torso_frame)
        #======#
        # Mass #
        #======#
        self.lower_leg_mass, self.upper_leg_mass, self.torso_mass = symbols('m_L, m_U, m_T')
        #=========#
        # Inertia #
        #=========#
        self.lower_leg_inertia, self.upper_leg_inertia, self.torso_inertia = symbols('I_Lz, I_Uz, I_Tz')
        self.lower_leg_inertia_dyadic = inertia(self.lower_leg_frame, 0, 0, self.lower_leg_inertia)
        self.lower_leg_central_inertia = (self.lower_leg_inertia_dyadic, self.lower_leg_mass_center)
        self.upper_leg_inertia_dyadic = inertia(self.upper_leg_frame, 0, 0, self.upper_leg_inertia)
        self.upper_leg_central_inertia = (self.upper_leg_inertia_dyadic, self.upper_leg_mass_center)
        self.torso_inertia_dyadic = inertia(self.torso_frame, 0, 0, self.torso_inertia)
        self.torso_central_inertia = (self.torso_inertia_dyadic, self.torso_mass_center)
        #==============#
        # Rigid Bodies #
        #==============#
        self.lower_leg = RigidBody('Lower Leg', self.lower_leg_mass_center, self.lower_leg_frame,
            self.lower_leg_mass, self.lower_leg_central_inertia)
        self.upper_leg = RigidBody('Upper Leg', self.upper_leg_mass_center, self.upper_leg_frame,
            self.upper_leg_mass, self.upper_leg_central_inertia)
        self.torso = RigidBody('Torso', self.torso_mass_center, self.torso_frame,
            self.torso_mass, self.torso_central_inertia)
        #=========#
        # Gravity #
        #=========#
        self.g = symbols('g')
        self.lower_leg_grav_force = (self.lower_leg_mass_center,
            -self.lower_leg_mass * self.g * self.inertial_frame.y)
        self.upper_leg_grav_force = (self.upper_leg_mass_center,
            -self.upper_leg_mass * self.g * self.inertial_frame.y)
        self.torso_grav_force = (self.torso_mass_center, -self.torso_mass * self.g * self.inertial_frame.y)
        #===============#
        # Joint Torques #
        #===============#
        self.ankle_torque, self.knee_torque, self.hip_torque = dynamicsymbols('T_a, T_k, T_h')
        self.lower_leg_torque = (self.lower_leg_frame,
            self.ankle_torque * self.inertial_frame.z - self.knee_torque *
            self.inertial_frame.z)
        self.upper_leg_torque = (self.upper_leg_frame,
            self.knee_torque * self.inertial_frame.z - self.hip_torque *
            self.inertial_frame.z)
        self.torso_torque = (self.torso_frame, self.hip_torque * self.inertial_frame.z)
        #=====================#
        # Equations of Motion #
        #=====================#
        self.coordinates = [self.theta1, self.theta2, self.theta3]
        self.speeds = [self.omega1, self.omega2, self.omega3]
        self.kane = KanesMethod(self.inertial_frame,
            self.coordinates,
            self.speeds,
            self.kinematical_differential_equations)
        self.loads = [self.lower_leg_grav_force,
            self.upper_leg_grav_force,
            self.torso_grav_force,
            self.lower_leg_torque,
            self.upper_leg_torque,
            self.torso_torque]
        self.bodies = [self.lower_leg, self.upper_leg, self.torso]
        self.fr, self.frstar = self.kane.kanes_equations(self.bodies, self.loads)
        self.mass_matrix = self.kane.mass_matrix_full
        self.forcing_vector = self.kane.forcing_full
        #=============================#
        # List the symbolic arguments #
        #=============================#
        #-----------#
        # Constants #
        #-----------#
        self.constants = [self.lower_leg_length,
            self.lower_leg_com_length,
            self.lower_leg_mass,
            self.lower_leg_inertia,
            self.upper_leg_length,
            self.upper_leg_com_length,
            self.upper_leg_mass,
            self.upper_leg_inertia,
            self.torso_com_length,
            self.torso_mass,
            self.torso_inertia,
            self.g]
        #--------------#
        # Time Varying #
        #--------------#
        self.coordinates = [self.theta1, self.theta2, self.theta3]
        self.speeds = [self.omega1, self.omega2, self.omega3]
        self.specified = [self.ankle_torque, self.knee_torque, self.hip_torque]
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
        self.numerical_constants = array([0.611,  # lower_leg_length [m]
            0.387,  # lower_leg_com_length [m]
            6.769,  # lower_leg_mass [kg]
            0.101,  # lower_leg_inertia [kg*m^2]
            0.424,  # upper_leg_length [m]
            0.193,  # upper_leg_com_length
            17.01,  # upper_leg_mass [kg]
            0.282,  # upper_leg_inertia [kg*m^2]
            0.305,  # torso_com_length [m]
            32.44,  # torso_mass [kg]
            1.485,  # torso_inertia [kg*m^2]
            9.81])  # acceleration due to gravity [m/s^2]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # self.x = zeros(6)
        # self.x[:3] = deg2rad(2.0)
        self.x = np.random.randn(6)
        return self._get_obs()

    def _get_obs(self):
        return self.x

    def sample_action(self):
        return np.random.randn(3)

    def step(self, action):
        self.x = odeint(self.right_hand_side, self.x, array([0., self.dt]),
            args=(action, self.numerical_constants))[1]
        cost = (self.x[0]**2 + self.x[1]**2 + self.x[2]**2 + .1*self.x[3]**2 + .1*self.x[4]**2 + .1*self.x[5]**2 + .001*action[0]**2 + .001*action[1]**2 + .001*action[2]**2)
        # print(x_temp)
        self.x[:3] = self.angle_normalize(self.x[:3])
        # self.x[:3] = (x_temp[:3]<0)*(2*np.pi+x_temp[:3]) + np.logical_and(((x_temp[:3]>=0), (x_temp[:3]<(2*np.pi))))*x_temp[:3] + (x_temp[:3]>(2*np.pi))*(x_temp[:3]-2*np.pi)
        # print(self.x[0:3])
        return self.x, -cost, False, {}

    def angle_normalize(self, angle_input):
        return (((angle_input+np.pi) % (2*np.pi)) - np.pi)

    def render(self, mode='human'):
        if not self.ax:
            fig, ax = plt.subplots()
            ax.set_xlim([-5, 5])
            ax.set_ylim([-5, 5])
            ax.set_aspect('equal')
            self.ax = ax
        else:
            self.ax.clear()
            self.ax.set_xlim([-5, 5])
            self.ax.set_ylim([-5, 5])
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
