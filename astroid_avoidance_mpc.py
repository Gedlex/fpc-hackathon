import casadi as ca
import control
from dataclasses import dataclass, field
import numpy as np
import os
import sys
import models.rocketXZModel as rocketXZModel
from models import RocketXZConfig

@dataclass
class RocketXZLQRCtrlConfig:
    Q: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0, 0.1, 0.1, 1.0, 0.1]))
    R: np.ndarray = field(default_factory=lambda: np.diag([0.1, 0.1]))
    x_equilibrium: np.ndarray = field(default_factory=lambda: np.array([0.3, 2.0, 0.0, 0.0, 0.0, 0.0]))

@dataclass
class RocketXZMPCCtrlConfig:
    Q: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0, 0.1, 0.1, 1.0, 0.1]))
    R: np.ndarray = field(default_factory=lambda: np.diag([0.1, 0.1]))
    x_equilibrium: np.ndarray = field(default_factory=lambda: np.array([0.3, 2.0, 0.0, 0.0, 0.0, 0.0]))

# Fix random seed and print options
np.random.seed(1)
np.set_printoptions(threshold=10000, linewidth=np.inf)

class RocketXZLQRCtrl:
    def __init__(self, sampling_time: float, model: rocketXZModel.RocketXZModel):
        self._sampling_time = sampling_time
        self._model = model
        self._ctrl_config = RocketXZLQRCtrlConfig()
        self._goal_val = None

        def compute_continuousT_feedback_gain(x_equilibrium: np.ndarray, u_equilibrium: np.ndarray):
            A, B = self._model.linearizeContinuousDynamics(x_equilibrium, u_equilibrium)
            Q = self._ctrl_config.Q
            R = self._ctrl_config.R
            K, S, E = control.lqr(A, B, Q, R)
            return K

        def compute_discreteT_feedback_gain(x_equilibrium: np.ndarray, u_equilibrium: np.ndarray):
            A, B = self._model.linearizeDiscreteDynamics(x_equilibrium, u_equilibrium)
            Q = self._ctrl_config.Q
            R = self._ctrl_config.R
            K, S, E = control.dlqr(A, B, Q, R)
            return K

        self.u_equilibrium = np.array([-self._model.model_config.mass*self._model.model_config.gravity, 0.0])
        self.fdbk_gain = compute_discreteT_feedback_gain(self._ctrl_config.x_equilibrium, self.u_equilibrium)


    def compute_LQR_control(self, x: np.ndarray,):
        x_error = x - self._ctrl_config.x_equilibrium
        u = self.u_equilibrium - self.fdbk_gain @ x_error
        return u

    @property
    def x_equilibrium(self):
        return self._ctrl_config.x_equilibrium
    


class RocketXZMPCCtrl:
    def __init__(self, sampling_time: float, model: rocketXZModel.RocketXZModel, horizon: int,
                 x_equilibrium: np.ndarray, u_equilibrium: np.ndarray):
        self._sampling_time = sampling_time
        self._model = model
        self._horizon = horizon
        self._x_equilibrium = x_equilibrium.reshape((-1, 1))
        self._u_equilibrium = u_equilibrium.reshape((-1, 1))
        self._ctrl_config = RocketXZLQRCtrlConfig()
        self._goal_val = None

        # initialize the MPC problem
        # init casadi Opti object which holds the optimization problem
        self.prob = ca.Opti()
        self.prob.solver('ipopt')

        # define optimization variables
        self.x = self.prob.variable(RocketXZConfig.nx, self._horizon+1)
        self.u = self.prob.variable(RocketXZConfig.nu, self._horizon)
        self.x_0 = self.prob.parameter(RocketXZConfig.nx)

        # define the objective
        objective = 0.0
        for i in range(self._horizon):
            objective += (self.x[:, i] - self._x_equilibrium).T @ self._ctrl_config.Q @ (self.x[:, i] - self._x_equilibrium) + (self.u[:, i] - self._u_equilibrium).T @ self._ctrl_config.R @ (self.u[:, i] - self._u_equilibrium)
        # objective += self.x[:, -1].T @ P @ self.x[:, -1] # accurate only when terminal state is close enough to origin!
        self.objective = objective
        self.prob.minimize(objective)

        # define the constraints
        self.constraints = [self.x[:, 0] == self.x_0]
        for i in range(self._horizon):
            # system dynamics equality
            self.constraints += [self.x[:, i+1] == model.f_disc(self.x[:, i], self.u[:, i])]

            # TODO: add more constraints here (e.g. asteroids)

            
        self.prob.subject_to(self.constraints)


    def compute_MPC_control(self, x: np.ndarray,):
        self.prob.set_value(self.x_0, x)
        sol = self.prob.solve()
        u = sol.value(self.u[:, 0])
        return u

    @property
    def x_equilibrium(self):
        return self._ctrl_config.x_equilibrium

sampling_time = 0.05
sim_length = 100
x_init = np.array([0., 1.0, 0.0, 0.0, 0., 0.])
model = rocketXZModel.RocketXZModel(sampling_time)
LQR = RocketXZLQRCtrl(sampling_time, model)
controller = RocketXZMPCCtrl(sampling_time, model, horizon=3, x_equilibrium=np.array([0.3, 2.0, 0.0, 0.0, 0.0, 0.0]), u_equilibrium=LQR.u_equilibrium)
x_trajectory, u_trajectory = model.simulateClosedLoop(sim_length, x_init, controller.compute_MPC_control)
additional_lines_or_scatters = {"Goal": {"type": "scatter", "data": [[controller.x_equilibrium[0]], [controller.x_equilibrium[1]]], "color": "tab:orange", "s": 100, "marker":"x"}}
model.animateSimulation(x_trajectory, u_trajectory, additional_lines_or_scatters=additional_lines_or_scatters)