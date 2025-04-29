import casadi as ca
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import numpy as np
from matplotlib import animation

from models.baseModel import BaseModel

"""
x = (px, pz, vx, vz, pitch, vpitch)
a = (T, delta)
"""

@dataclass
class RocketXZConfig:
    nx: int = 6
    nu: int = 2
    d: float = 0.8
    mass: float = 0.7
    inertia: float = 0.3
    gravity: float = 9.81


class RocketXZModel(BaseModel):
    def __init__(self, sampling_time, gravity=True):
        super().__init__(sampling_time)
        self.model_name = "RocketXZModel"
        self.model_config = RocketXZConfig()
        if not gravity:
            self.model_config.gravity = 0.0

        x = ca.MX.sym('x', self.model_config.nx)
        u = ca.MX.sym('u', self.model_config.nu)
        x_dot = ca.vertcat(
            x[2],  # \dot{px}
            x[3],  # \dot{pz}
            u[0] * ca.sin(x[4] + u[1]) / self.model_config.mass,  # \dot{vx}
            -self.model_config.gravity - u[0] * ca.cos(x[4] + u[1]) / self.model_config.mass,  # \dot{vz}
            x[5],  # \dot{pitch}
            -u[0] * ca.sin(u[1]) * self.model_config.d / self.model_config.inertia  # \dot{vpitch}
        )
        dae = {'x': x, 'p': u, 'ode': x_dot}
        opts = {'tf': self._sampling_time}

        self.I = ca.integrator('I', 'rk', dae, opts)

        self.A_func = ca.Function('A_func', [x, u], [ca.jacobian(x_dot, x)])
        self.B_func = ca.Function('B_func', [x, u], [ca.jacobian(x_dot, u)])

        self.A_disc_func = ca.Function('A_disc_func', [x, u], [ca.jacobian(self.I(x0=x, p=u)['xf'], x)])
        self.B_disc_func = ca.Function('B_disc_func', [x, u], [ca.jacobian(self.I(x0=x, p=u)['xf'], u)])

        # create continuous and discrete dynamics
        self.f_cont = ca.Function('f_cont', [x, u], [x_dot])
        self.f_disc = ca.Function('f_disc', [x, u], [self.I(x0=x, p=u)['xf']])

    def linearizeContinuousDynamics(self, x, u):
        A = self.A_func(x, u).full()
        B = self.B_func(x, u).full()
        return A, B


    def linearizeDiscreteDynamics(self, x, u):
        A = self.A_disc_func(x, u).full()
        B = self.B_disc_func(x, u).full()
        return A, B


    def animateSimulation(self, x_trajectory, u_trajectory, additional_lines_or_scatters=None):
        def compute_rocket_vertices(x:np.ndarray):
            rocket_width = 0.2
            rot_matrix = np.array([[np.cos(x[4]), -np.sin(x[4])], [np.sin(x[4]), np.cos(x[4])]])
            vertices = np.array([[rocket_width, -rocket_width, -rocket_width, rocket_width, rocket_width], [self.model_config.d, self.model_config.d, -self.model_config.d, -self.model_config.d, self.model_config.d], ])
            vertices = np.concatenate((vertices, np.array([[0.], [self.model_config.d*1.3]]), np.array([[-rocket_width], [self.model_config.d]])), axis=1)
            vertices = rot_matrix @ vertices + x[:2].reshape(2, 1)
            return vertices
        def compute_thrust_polygon(x:np.ndarray, u:np.ndarray):
            thrust_width = 0.1
            thrust_scaling = -0.05
            vertices = np.array([[0.0, thrust_width, -thrust_width], [0.0, -u[0], -u[0]]])
            vertices[1, :] *= thrust_scaling
            rot_matrix_1 = np.array([[np.cos(u[1]), -np.sin(u[1])], [np.sin(u[1]), np.cos(u[1])]])
            vertices = rot_matrix_1 @ vertices + np.array([[0.], [-self.model_config.d]])
            rot_matrix_2 = np.array([[np.cos(x[4]), -np.sin(x[4])], [np.sin(x[4]), np.cos(x[4])]])
            vertices = rot_matrix_2 @ vertices + x[:2].reshape(2, 1)
            polygon = patches.Polygon(vertices.T, closed=True, color="tab:red", alpha=0.5, label="Thrust")
            return polygon

        fontsize = 16
        params = {
            'text.latex.preamble': r"\usepackage{gensymb} \usepackage{amsmath} \usepackage{amsfonts} \usepackage{cmbright}",
            'axes.labelsize': fontsize,
            'axes.titlesize': fontsize,
            'legend.fontsize': fontsize,
            'xtick.labelsize': fontsize,
            'ytick.labelsize': fontsize,
            "mathtext.fontset": "stixsans",
            "axes.unicode_minus": False,
        }
        matplotlib.rcParams.update(params)

        sim_length = u_trajectory.shape[1]
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, 5)
        ax.set_xlabel('px(m)', fontsize=14)
        ax.set_ylabel('pz(m)', fontsize=14)

        rocket_line, = ax.plot([], [], color="tab:blue", linewidth=2, zorder=1)
        rocket_dot = ax.scatter([], [], color="tab:gray", s=50, zorder=2)
        thrust_patch = [None]  # mutable container for patch
        extra_elements = []

        if additional_lines_or_scatters:
            for key, value in additional_lines_or_scatters.items():
                if value["type"] == "scatter":
                    s = ax.scatter(value["data"][0], value["data"][1], color=value["color"], s=value["s"], label=key, marker=value["marker"], zorder=3)
                    extra_elements.append(s)
                elif value["type"] == "line":
                    l, = ax.plot(value["data"][0], value["data"][1], color=value["color"], linewidth=2, label=key)
                    extra_elements.append(l)
            ax.legend(loc="lower right")

        def init():
            rocket_line.set_data([], [])
            rocket_dot.set_offsets(np.empty((0, 2)))
            return [rocket_line, rocket_dot] + extra_elements

        def update(i):
            ax.set_title(f"Rocket XZ Simulation: Step {i+1}")
            x = x_trajectory[:, i]
            u = u_trajectory[:, i] if i < sim_length else u_trajectory[:, -1]
            rocket_vertices = compute_rocket_vertices(x)
            rocket_line.set_data(rocket_vertices[0, :], rocket_vertices[1, :])
            rocket_dot.set_offsets([[x[0], x[1]]])

            # Remove old patch if exists
            if thrust_patch[0]:
                thrust_patch[0].remove()
            thrust_patch[0] = compute_thrust_polygon(x, u)
            ax.add_patch(thrust_patch[0])
            return [rocket_line, rocket_dot, thrust_patch[0]] + extra_elements

        anim = animation.FuncAnimation(fig, update, frames=sim_length+1, init_func=init, blit=False, repeat=False)
        return anim 