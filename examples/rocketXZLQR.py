import casadi as ca
import control
from dataclasses import dataclass, field
import numpy as np
import os
import sys

local_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(local_path, ".."))

import models.rocketXZModel as rocketXZModel


@dataclass
class RocketXZLQRCtrlConfig:
    Q: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0, 0.1, 0.1, 1.0, 0.1]))
    R: np.ndarray = field(default_factory=lambda: np.diag([0.1, 0.1]))
    x_equilibrium: np.ndarray = field(default_factory=lambda: np.array([0.3, 2.0, 0.0, 0.0, 0.0, 0.0]))


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


def main():
    sampling_time = 0.05
    sim_length = 100
    x_init = np.array([0., 1.0, 0.0, 0.0, 0., 0.])
    model = rocketXZModel.RocketXZModel(sampling_time)
    controller = RocketXZLQRCtrl(sampling_time, model)
    x_trajectory, u_trajectory = model.simulateClosedLoop(sim_length, x_init, controller.compute_LQR_control)
    additional_lines_or_scatters = {"Goal": {"type": "scatter", "data": [[controller.x_equilibrium[0]], [controller.x_equilibrium[1]]], "color": "tab:orange", "s": 100, "marker":"x"}}
    model.animateSimulation(x_trajectory, u_trajectory, additional_lines_or_scatters=additional_lines_or_scatters)
    return


if __name__ == "__main__":
    main()