import casadi as ca
from dataclasses import dataclass, field
import numpy as np
import os
import sys

local_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(local_path, ".."))

import models.omniBotXYModel as omniBotXYModel


@dataclass
class RefTrackingCtrlConfig:
    Q: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0, 0.01, 0.01]))
    R: np.ndarray = field(default_factory=lambda: np.diag([0.01, 0.01]))
    n_hrzn = 50
    num_agents = 2


class OmniBotXYMultiAgentCtrl:
    def __init__(self, sampling_time: float, num_agents:int, model: omniBotXYModel.OmniBotXYModel):
        self._sampling_time = sampling_time
        self._num_agents = num_agents
        self._model = model
        self._ctrl_config = RefTrackingCtrlConfig()
        self.x_sol = None
        self.u_sol = None
        self._x_ref = None

    def _define_ocp_variables(self):
        self._x_opt = ca.MX.sym('x', self._model.model_config.nx * self._num_agents, self._ctrl_config.n_hrzn + 1, )
        self._x_ref = ca.MX.sym('x_ref',  self._model.model_config.nx * self._num_agents, self._ctrl_config.n_hrzn + 1)
        self._u_opt = ca.MX.sym('u', self._model.model_config.nu * self._num_agents, self._ctrl_config.n_hrzn, )
        self._x_0_param = ca.MX.sym('x_0', self._model.model_config.nx * self._num_agents)

    def _setup_constraints(self):
        self._g = []
        self._lbg = []
        self._ubg = []
        nx = self._model.model_config.nx
        nu = self._model.model_config.nu
        # System dynamics
        self._g.append(self._x_opt[:self._num_agents*nx, 0] - self._x_0_param)
        self._lbg.append(np.zeros(self._num_agents * nx,))
        self._ubg.append(np.zeros(self._num_agents * nx,))
        for i_agent in range(self._num_agents):
            for i in range(self._ctrl_config.n_hrzn):
                self._g.append(self._x_opt[i_agent*nx:(i_agent+1)*nx, i+1] - self._model.f_disc(self._x_opt[i_agent*nx:(i_agent+1)*nx, i], self._u_opt[i_agent*nu:(i_agent+1)*nu, i]))
        self._lbg.append(np.zeros(nx*self._ctrl_config.n_hrzn*self._num_agents,))
        self._ubg.append(np.zeros(nx*self._ctrl_config.n_hrzn*self._num_agents,))
        # Collision Avoidance Constraints
        for i_agent in range(self._num_agents):
            for j_agent in range(i_agent+1, self._num_agents):
                for i in range(1, self._ctrl_config.n_hrzn+1):
                    self._g.append(ca.sumsqr(self._x_opt[i_agent*nx:i_agent*nx+2, i] - self._x_opt[j_agent*nx:j_agent*nx+2, i]))
                    self._lbg.append((self._model.model_config.safety_radius * 2)**2)
                    self._ubg.append(ca.inf)
        return

    def _setup_obj_func(self):
        nx = self._model.model_config.nx
        nu = self._model.model_config.nu
        self._J = 0.0
        for i_agent in range(self._num_agents):
            for i in range(self._ctrl_config.n_hrzn):
                x_dev = self._x_opt[i_agent*nx:(i_agent+1)*nx, i] - self._x_ref[i_agent*nx:(i_agent+1)*nx, i]
                self._J += x_dev.T @ self._ctrl_config.Q @ x_dev
                u = self._u_opt[i_agent*nu:(i_agent+1)*nu, i]
                self._J += u.T @ self._ctrl_config.R @ u
            # Terminal cost
            x_dev = self._x_opt[i_agent*nx:(i_agent+1)*nx, -1] - self._x_ref[i_agent*nx:(i_agent+1)*nx, -1]
            self._J += x_dev.T @ self._ctrl_config.Q @ x_dev
        return

    def setup_OCP(self):
        self._define_ocp_variables()
        self._setup_constraints()
        self._setup_obj_func()
        ocp = {
            'x': ca.veccat(self._x_opt, self._u_opt),
            'p': ca.veccat(self._x_0_param, self._x_ref),
            'g': ca.vertcat(*self._g),
            'f': self._J
        }
        opts = {'ipopt': {'print_level': 0, 'max_iter': 1000}, "print_time": False}
        self._solver = ca.nlpsol('solver', 'ipopt', ocp, opts)
        return

    def solve_OCP(self, x_0: np.ndarray):
        nx = self._model.model_config.nx
        nu = self._model.model_config.nu
        x_0 = x_0.flatten()[:, np.newaxis]
        assert x_0.shape == (nx * self._num_agents, 1), f"Expected shape {(nx * self._num_agents, 1)}, but got {x_0.shape}"
        if self.x_sol is None:
            self.x_sol = np.tile(x_0, (1, self._ctrl_config.n_hrzn + 1))
        if self.u_sol is None:
            self.u_sol = np.zeros((nu * self._num_agents, self._ctrl_config.n_hrzn))
        if self._x_ref is None:
            raise ValueError("Ref value must be set before solving the OCP.")

        solution = self._solver(
            x0=ca.veccat(self.x_sol, self.u_sol),
            p=ca.veccat(x_0, self._x_ref),
            lbg=ca.vertcat(*self._lbg),
            ubg=ca.vertcat(*self._ubg)
        )
        x_sol = solution['x'].full().flatten()[:(self._ctrl_config.n_hrzn+1) * nx * self._num_agents].reshape((nx * self._num_agents, self._ctrl_config.n_hrzn+1), order='F')
        u_sol = solution['x'].full().flatten()[(self._ctrl_config.n_hrzn+1) * nx * self._num_agents:].reshape((nu * self._num_agents, self._ctrl_config.n_hrzn, ), order='F')
        self.x_sol = x_sol
        self.u_sol = u_sol

        return u_sol[:, 0]

    @property
    def n_hrzn(self):
        return self._ctrl_config.n_hrzn

    @property
    def x_ref(self):
        return self._x_ref

    @x_ref.setter
    def x_ref(self, x_ref: np.ndarray):
        if x_ref.shape != (self._model.model_config.nx * self._num_agents, self._ctrl_config.n_hrzn + 1):
            raise ValueError(f"Expected shape {(self._model.model_config.nx * self._num_agents, self._ctrl_config.n_hrzn + 1)}, but got {x_ref.shape}")
        self._x_ref = x_ref.copy()



def open_loop():
    sampling_time = 0.05
    num_agents = 2
    p1_target_val = np.array([3.5, 2.5])
    x1_init_val = np.array([0.0, 0.0, 0.0, 0.0])
    p2_target_val = np.array([0., 2.5])
    x2_init_val = np.array([2.5, 0.7, 0.0, 0.0])

    model = omniBotXYModel.OmniBotXYModel(sampling_time)
    controller = OmniBotXYMultiAgentCtrl(sampling_time, num_agents=num_agents, model=model)
    controller.setup_OCP()

    p1_ref_val = (p1_target_val[:, np.newaxis] - x1_init_val[:2, np.newaxis]) @ np.linspace(0.0, 1.0, controller.n_hrzn+1, endpoint=True)[np.newaxis, :] + x1_init_val[:2, np.newaxis]
    v1_ref_val = np.diff(p1_ref_val, axis=1) / sampling_time
    v1_ref_val = np.hstack((v1_ref_val, np.zeros((2, 1))))

    p2_ref_val = (p2_target_val[:, np.newaxis] - x2_init_val[:2, np.newaxis]) @ np.linspace(0.0, 1.0, controller.n_hrzn+1, endpoint=True)[np.newaxis, :] + x2_init_val[:2, np.newaxis]
    v2_ref_val = np.diff(p2_ref_val, axis=1) / sampling_time
    v2_ref_val = np.hstack((v2_ref_val, np.zeros((2, 1))))
    x_ref_val = np.vstack((p1_ref_val, v1_ref_val, p2_ref_val, v2_ref_val))

    controller.x_ref = x_ref_val
    controller.solve_OCP(np.hstack((x1_init_val, x2_init_val)))
    u_trajectory = controller.u_sol
    x_trajectory = controller.x_sol

    additional_lines_or_scatters = {"Ref1": {"type": "line", "data": [p1_ref_val[0, :], p1_ref_val[1, :]], "color": "tab:orange", "s": 100, "marker":"x"},
                                    "Ref2": {"type": "line", "data": [p2_ref_val[0, :], p2_ref_val[1, :]], "color": "tab:orange", "s": 100, "marker":"x"}}
    model.animateSimulation(x_trajectory, u_trajectory, num_agents=num_agents, additional_lines_or_scatters=additional_lines_or_scatters)


if __name__ == "__main__":
    open_loop()