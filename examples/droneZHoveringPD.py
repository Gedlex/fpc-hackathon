import casadi as ca
from dataclasses import dataclass
import numpy as np
import os
import sys

local_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(local_path, ".."))

import models.drone1DModel as droneZModel


@dataclass
class HoveringPDCtrlConfig:
    max_thrust: float = 30.0
    min_thrust: float = 0.0
    PD_Kp: float = 2.0
    PD_Kd: float = 1.0


class DroneZHoveringPDCtrl:
    """
    f (t) = -Kp * (z - target) - Kd * vz
    """
    def __init__(self, sampling_time: float, model: droneZModel.DroneZModel):
        self._sampling_time = sampling_time
        self._model = model
        self._ctrl_config = HoveringPDCtrlConfig()
        self._goal_val = None
        self._error_accumulated = 0.0


    def compute_PD_control(self, x: np.ndarray):
        z_error = x[0] - self._goal_val
        vz = x[1]
        self._error_accumulated += z_error * self._sampling_time
        thrust = self._model.model_config.mass * self._model.model_config.gravity - self._ctrl_config.PD_Kp * z_error  - self._ctrl_config.PD_Kd * vz

        if thrust >= 1e-4 + self._ctrl_config.max_thrust or thrust <= -1e-4 - self._ctrl_config.min_thrust:
            print(f"Warning: thrust {thrust} is out of bounds [{self._ctrl_config.min_thrust}, {self._ctrl_config.max_thrust}]")
        thrust = np.clip(thrust, self._ctrl_config.min_thrust, self._ctrl_config.max_thrust)
        return np.array([thrust, ])


    @property
    def goal(self):
        return self._goal_val

    @goal.setter
    def goal(self, value):
        self._goal_val = value
        self._error_accumulated = 0.0


def main():
    sampling_time = 0.05
    sim_length = 200
    x_init = np.array([0.02, 0.0])
    goal = 2.0
    model = droneZModel.DroneZModel(sampling_time)
    controller = DroneZHoveringPDCtrl(sampling_time, model)
    controller.goal = goal
    x_trajectory, u_trajectory = model.simulateClosedLoop(sim_length, x_init, controller.compute_PD_control)
    additional_lines_or_scatters = {"Goal": {"type": "scatter", "data": [[0.], [goal]], "color": "tab:orange", "s": 100, "marker":"x", "idx_ax": 0}, "Goal_line": {"type": "line", "data": [[0., sim_length*sampling_time], [goal, goal]], "color": "tab:orange", "s": 100, "marker":"x", "idx_ax": 1}}
    model.animateSimulation(x_trajectory, u_trajectory, additional_lines_or_scatters=additional_lines_or_scatters)
    return


if __name__ == "__main__":
    main()