from collections.abc import Callable
import casadi as ca
import numpy as np

class BaseModel:
    model_name = "BaseModel"
    model_config = None
    _sampling_time = np.nan

    def __init__(self, sampling_time: float):
        self._sampling_time = sampling_time


        self.f_cont: ca.Function = None
        """ Casadi Function for the rhs f(x,u) of the continuous dynamics \dot x = f(x,u). Use .full() to cast the result to a numpy array. """
        self.f_disc: ca.Function = None
        """ Casadi Function for the rhs f(x,u) of the discrete dynamics x_{k+1} = F(x_k, u_k) 
        from an RK4 integrator, which use the piecewise constant control u_k over the interval specified by 'sampling time'.
          Use .full() to cast the result to a numpy array. """

    def animateSimulation(self, x_trajectory: np.ndarray, u_trajectory: np.ndarray):
        raise NotImplementedError("Subclasses must implement animateSimulation.")

    def define_constraint(x: np.ndarray, asteroid: np.ndarray):
        x_asteroid = asteroid[0]
        z_asteroid = asteroid[1]
        x_rocket = x[0]
        z_rocket = x[1]
        radius_asteroid = asteroid[2]
        radius_rocket = 1
        # Define the constraint function
        dist_squared = (x_rocket - x_asteroid)**2 + (z_rocket - z_asteroid)**2
        min_dist_squared = (radius_asteroid + radius_rocket)**2
        return min_dist_squared - dist_squared <= 0



    def createAsteroid(self, x: np.ndarray):
        x_asteroid = x[0]
        z_asteroid = x[1] + 2
        radius_asteroid = 0.5
        return np.array([x_asteroid, z_asteroid, radius_asteroid])

    def simulateClosedLoop(self, sim_length:int, x_init:np.ndarray, controller: Callable, num_agents:int=1):
        nx = self.model_config.nx
        nu = self.model_config.nu
        x_trajectory = np.zeros((nx * num_agents, sim_length+1))
        u_trajectory = np.zeros((nu * num_agents, sim_length))
        x_trajectory[:, 0] = x_init.flatten()
        asteroid_coords1 = self.createAsteroid(x_init[:2] + np.array([1, 0]))
        asteroid_coords2 = self.createAsteroid(np.array([4, 0]))
        asteroid_coords = np.vstack((asteroid_coords1, asteroid_coords2))
        for i in range(sim_length):
            u = controller(x_trajectory[:, i], asteroid_coords)
            u_trajectory[:, i] = u
            for i_agent in range(num_agents):
                x_trajectory[i_agent*nx:(i_agent+1)*nx, i+1] = self.f_disc(x_trajectory[i_agent*nx:(i_agent+1)*nx, i], u[i_agent*nu:(i_agent+1)*nu]).full().flatten()
            print(f"Step {i+1}: x = {x_trajectory[:, i+1]}, u = {u}")
        return x_trajectory, u_trajectory, asteroid_coords


    def simulateOpenLoop(self, x_init: np.ndarray, u_trajectory: np.ndarray, num_agents:int=1):
        sim_length = u_trajectory.shape[1]
        nx = self.model_config.nx
        nu = self.model_config.nu
        x_trajectory = np.zeros((nx * num_agents, sim_length + 1))
        x_trajectory[:, 0] = x_init.flatten()
        for i in range(sim_length):
            for i_agent in range(num_agents):
                x_trajectory[i_agent*nx:(i_agent+1)*nx, i+1] = self.f_disc(x_trajectory[i_agent*nx:(i_agent+1)*nx, i], u_trajectory[i_agent*nu:(i_agent+1)*nu, i]).full().flatten()
            print(f"Step {i+1}: x = {x_trajectory[:, i+1]}, u = {u_trajectory[:, i]}")
        return x_trajectory