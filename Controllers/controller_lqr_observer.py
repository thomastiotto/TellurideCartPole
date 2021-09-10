"""
This is a linear-quadratic regulator
It assumes that the input relation is u = Q*u_max (no fancy motor model) !
"""

import scipy
import numpy as np
from scipy import signal

from datetime import datetime
from numpy.random import SFC64, Generator

from Controllers.template_controller import template_controller
from CartPole.state_utilities import create_cartpole_state, cartpole_state_varname_to_index
from CartPole.cartpole_model import u_max, s0
from CartPole.cartpole_jacobian import cartpole_jacobian

import yaml
config = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)
Q = np.diag(config["controller"]["lqr"]["Q"])
R = config["controller"]["lqr"]["R"]

Ts = 0.02  # sampling frequency in sec

# MODE = 'Continuous'
MODE = 'Discrete'

# DISCRETISATION = 'Euler'
DISCRETISATION = 'Exact'

OBSERVER = 'Luenberger'
# OBSERVER = 'Euler + IIR Filter'

pole_multiplier = 8.0

#  Noise:
sigma_position = 0.005
sigma_angle = 0.003

# Variables for 'Euler + IIR Filter'
angle_smoothing = 0.6
position_smoothing = 0.6
angleD_smoothing = 0.8
positionD_smoothing = 0.8

def calculate_minimal_robust_invariant_set (A_closed, Kf, C, w_max, tol):

    i = 1

    e_max = np.array([
        [0.0],  # position error
        [0.0],
        [0.0],
        [0.0]# angle error
    ])

    v_max = abs(Kf @ w_max)

    A_closed_f = A_closed - Kf @ C
    A_closed_f_n = np.copy(A_closed_f)


    e_max_next = v_max

    while np.any(abs(e_max_next - e_max)) > tol:
        i += 1
        e_max = e_max_next
        A_closed_f_n = A_closed_f_n @ A_closed_f
        e_max_next = abs(e_max)+abs(A_closed_f_n @ v_max)


    e_relative = np.array([
        [abs(e_max_next[0,0]/w_max[0,0])],
        [abs(e_max_next[0, 0] / w_max[0, 0])],
        [abs(e_max_next[0, 0] / w_max[1, 0])],
        [abs(e_max_next[0, 0] / w_max[1, 0])]
    ])

    return e_max_next, e_relative, i


class controller_lqr_observer(template_controller):
    def __init__(self):
        # From https://github.com/markwmuller/controlpy/blob/master/controlpy/synthesis.py#L8
        """Solve the continuous time LQR controller for a continuous time system.

        A and B are system matrices, describing the systems dynamics:
         dx/dt = A x + B u

        The controller minimizes the infinite horizon quadratic cost function:
         cost = integral (x.T*Q*x + u.T*R*u) dt

        where Q is a positive semidefinite matrix, and R is positive definite matrix.

        Returns K, P, eigVals:
        Returns gain the optimal gain K, the solution matrix P, and the closed loop system eigenvalues.
        The optimal input is then computed as:
         input: u = -K*x
        """
        # ref Bertsekas, p.151

        print('--------------------------------------------------')
        print('LQR')
        print('MODE: {}'.format(MODE))
        if MODE == 'Discrete':
            print('DISCRETIZATION: {}'.format(DISCRETISATION))
        print('OBSERVER: {}'.format(OBSERVER))
        print('--------------------------------------------------')

        self.controller_data_for_csv = {
                                        'angle_estimate': [0.0],
                                        'angleD_estimate': [0.0],
                                        'position_estimate': [0.0],
                                        'positionD_estimate': [0.0],
                                        'angle_measurement': [0.0],
                                        'position_measurement': [0.0],
                                        }

        SEED = int((datetime.now() - datetime(1970, 1, 1)).total_seconds() * 1000.0)
        self.rng_lqr_observer = Generator(SFC64(SEED))

        # Calculate Jacobian around equilibrium
        # Set point around which the Jacobian should be linearized
        # It can be here either pole up (all zeros) or pole down
        s = s0
        s[cartpole_state_varname_to_index('position')] = 0.0
        s[cartpole_state_varname_to_index('positionD')] = 0.0
        s[cartpole_state_varname_to_index('angle')] = 0.0
        s[cartpole_state_varname_to_index('angleD')] = 0.0
        u = 0.0

        jacobian = cartpole_jacobian(s, u)
        self.A = jacobian[:, :-1]
        self.B = np.reshape(jacobian[:, -1], newshape=(4, 1)) * u_max
        self.C = np.array([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0]])
        self.D = np.array([[0.0],
                           [0.0]])
        if DISCRETISATION == 'Euler':
            self.Ad = np.eye(4) + Ts * self.A
            self.Bd = Ts * self.B
            self.Cd = self.C
            self.Dd = self.D
        elif DISCRETISATION == 'Exact':
            self.Ad, self.Bd, self.Cd, self.Dd, _ = scipy.signal.cont2discrete((self.A, self.B, self.C, self.D), dt=Ts, method='zoh')

        # Cost matrices for LQR controller
        self.Q = Q  # How much to punish x, v, theta, omega
        self.R = R  # How much to punish Q


        # Solve the ricatti equation
        P = scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
        # compute the LQR gain
        if np.array(self.R).ndim == 0:
            Ri = 1.0 / self.R
        else:
            Ri = np.linalg.inv(self.R)

        self.K = np.dot(Ri, (np.dot(self.B.T, P)))

        # It is not necessary to solve are separately for discrete case
        # K=Kd and poles_system_d = exp(Ts*poles_system)
        Pd = scipy.linalg.solve_discrete_are(self.Ad, self.Bd, self.Q, self.R)
        self.Kd = np.linalg.inv(self.R+self.Bd.T@Pd@self.Bd)@self.Bd.T@Pd@self.Ad

        self.A_closed = self.A-self.B@self.K # State matrix for closed loop system under controller K
        self.Ad_closed = self.Ad-self.Bd@self.Kd

        poles_system = np.linalg.eigvals(self.A_closed)

        # Poles for filter
        poles_filter = poles_system*pole_multiplier
        poles_filter_d = np.exp(poles_filter*Ts)
        # # Alternative way of setting poles
        # slowest_pole_ct = poles_system[np.argmax(poles_system.real)]
        # pole1_ct = slowest_pole_ct * 10
        # pole2_ct = np.conj(pole1_ct)
        # poles_obs_ct = np.array([pole1_ct, pole1_ct-1, pole2_ct, pole2_ct-1])
        # print('poles obs ct', poles_obs_ct)

        # Calculate filter matrix
        obs = signal.place_poles(self.A_closed.T, self.C.T, poles_filter)
        self.Kf = obs.gain_matrix.T

        obs_d = signal.place_poles(self.Ad_closed.T, self.Cd.T, poles_filter_d, rtol=1.0e-7, maxiter=100)
        self.Kfd = obs_d.gain_matrix.T


        # Calculate noise boundary Ε (as in tube MPC)
        tol = 1.0e-7
        w_max = np.array([
            [3.0 * sigma_position],  # position error
            [3.0 * sigma_angle],  # angle error
        ])

        # The method from MPC lecture for mRIS does work only for discrete system!
        # E, E_relative, i = calculate_minimal_robust_invariant_set(self.A_closed, self.Kf, self.C, w_max, tol)
        Ed, Ed_relative, i_d = calculate_minimal_robust_invariant_set(A_closed=self.Ad_closed, Kf=self.Kfd, C=self.Cd, w_max=w_max, tol=tol)

        print('----------------------------------')
        print('Discrete:')
        print("Ed: {}".format(Ed))
        print("Ed_rel: {}".format(Ed_relative))
        print("i_d: {}".format(i_d))
        print('----------------------------------')

        self.state_estimate = np.array(
            [
                [0.0],
                [0.0],
                [0.0],
                [0.0]
            ]
        )

        self.initialize_estimator = True

        self.next_state_estimate_centered = np.copy(self.state_estimate)
        self.state_derivative = np.copy(self.state_estimate)

        # For filter as it is now in physical cartpole
        self.previous_state_estimate = np.copy(self.state_estimate)

    def step(self, s: np.ndarray, target_position: np.ndarray, time=None):

        # True state - you are not allowed to use it!
        state_true = np.array(
            [
                [s[cartpole_state_varname_to_index('position')]],
                [s[cartpole_state_varname_to_index('positionD')]],
                [s[cartpole_state_varname_to_index('angle')]],
                [s[cartpole_state_varname_to_index('angleD')]]
            ]
        )

        # Output_measurement is what you receive at every step
        output_measurement = np.array(
            [
                [(s[cartpole_state_varname_to_index('position')]) + sigma_position*self.rng_lqr_observer.standard_normal()],
                [s[cartpole_state_varname_to_index('angle')] + sigma_angle*self.rng_lqr_observer.standard_normal()],
            ]
        )

        if OBSERVER == 'Euler + IIR Filter':
            # Estimation of the state by euler integration and filtering

            self.state_estimate[2, 0] = output_measurement[1, 0] * (angle_smoothing) + (1 - angle_smoothing) * self.previous_state_estimate[2, 0]  # filter angle
            self.state_estimate[0, 0] = output_measurement[0, 0] * (position_smoothing) + (1 - position_smoothing) * self.previous_state_estimate[0, 0]  #filter position

            self.state_estimate[3, 0] = (self.state_estimate[2, 0] - self.previous_state_estimate[2, 0]) / Ts  # angleD
            self.state_estimate[1, 0] = (self.state_estimate[0, 0] - self.previous_state_estimate[0, 0]) / Ts  # positionD

            self.state_estimate[3, 0] = self.state_estimate[3, 0] * (angleD_smoothing) + (1 - angleD_smoothing) * self.previous_state_estimate[3, 0]  # Filter angleD
            self.state_estimate[1, 0] = self.state_estimate[1, 0] * (positionD_smoothing) + (1 - positionD_smoothing) * self.previous_state_estimate[1, 0]  # Filter positionD

            self.previous_state_estimate[...] = self.state_estimate[...]

            state_estimate_centered = np.copy(self.state_estimate)
            state_estimate_centered[0, 0] -= target_position


        elif OBSERVER == 'Luenberger':
            # self.state_estimate = state_true

            if self.initialize_estimator is True:
                self.initialize_estimator = False
                self.state_estimate[0, 0] = output_measurement[0, 0]
                self.state_estimate[2, 0] = output_measurement[1, 0]
                self.next_state_estimate_centered[...] = self.state_estimate[...]
                self.next_state_estimate_centered[0] -= target_position
            else:
                self.state_estimate[...] = self.next_state_estimate_centered[...]
                self.state_estimate[0, 0] += target_position
            state_true_centered = np.copy(state_true)
            state_true_centered[0, 0] -= target_position
            state_estimate_centered = np.copy(self.next_state_estimate_centered)

            output_estimate = np.copy(self.state_estimate[(0, 2), :])  # y = C*x

            if MODE == 'Continuous':
                # Notice that state_estimate_centered is a good choice because v is not calculated as an approximate derivative, rather it is a mapping state->derivative
                self.state_derivative = self.A_closed @ state_estimate_centered + self.Kf @ (output_measurement - output_estimate)
                self.next_state_estimate_centered = state_estimate_centered + Ts*self.state_derivative

            elif MODE == 'Discrete':
                self.next_state_estimate_centered = self.Ad_closed @ state_estimate_centered  + self.Kfd @ (output_measurement - output_estimate)




        # Q should be calculated based on just the state_estimate(centered)
        # You can use state_true(centered) for reference solution
        Q = np.asscalar(np.dot(-self.K, state_true_centered))
        # Q = np.asscalar(np.dot(-self.K, state_estimate_centered))

        # Clip Q
        if Q > 1.0:
            Q = 1.0
        elif Q < -1.0:
            Q = -1.0
        else:
            pass

        self.controller_data_for_csv = {
                                        'position_estimate': [self.state_estimate[0, 0]],
                                        'positionD_estimate': [self.state_estimate[1, 0]],
                                        'angle_estimate': [self.state_estimate[2, 0]],
                                        'angleD_estimate': [self.state_estimate[3, 0]],
                                        'position_measurement': [output_measurement[0, 0]],
                                        'angle_measurement': [output_measurement[1, 0]],
                                        }

        return Q

    def controller_reset(self):
        self.initialize_estimator = True

        self.state_estimate = np.array(
            [
                [0.0],
                [0.0],
                [0.0],
                [0.0]
            ]
        )

        self.next_state_estimate_centered = np.copy(self.state_estimate)
        self.state_derivative = np.copy(self.state_estimate)
        self.previous_state_estimate = np.copy(self.state_estimate)


if __name__ == "__main__":

    cont = controller_lqr_observer()

    # Set non-zero input
    s = s0
    s[cartpole_state_varname_to_index('position')-2] = -30.2
    s[cartpole_state_varname_to_index('positionD')-2] = 2.87
    s[cartpole_state_varname_to_index('angle')] = -0.32
    s[cartpole_state_varname_to_index('angleD')] = 0.237

    u = -0.24
