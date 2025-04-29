# (Model) 2D-XZ Drone Control
*Future PhD workshop, Hannover, April 2025, Jakob Harzer, Yunfan Gao, Moritz Diehl*

In this model we consider a drone in the 2D xz-plane:
<img src="_misc/2DroneImage.png" width="350"/>

## Dynamics

$$
\begin{aligned}
x = \begin{bmatrix}
p \\
v \\
\phi \\
\dot{\phi}
\end{bmatrix} = \begin{bmatrix}
p_x \\
p_z \\
v_x \\
v_z \\
\phi \\
\dot{\phi}
\end{bmatrix}\in \mathrm{R}^6 && u = \begin{bmatrix}
u_\mathrm{l} \\
u_\mathrm{r}
\end{bmatrix} \in \mathrm{R}^2
\end{aligned}
$$

with 2D position $p \in \mathrm{R}^2$, velocity $v \in \mathrm{R}^2$, orientation angle $\phi$ relative to the vertical axis and rotational velocity $\dot{\phi}$, $\dot{\phi} >0$ is a counterclockwise rotation. The drone is controlled using two positive and bounded rotor forces $0 \leq (u_\mathrm{l},u_\mathrm{r}) \leq u_\mathrm{max}$.

The following forces act on the drone:
- Propellors forces $u_\mathrm{l}, u_\mathrm{r}$ given as controls at distance $d = 5\,\mathrm{cm}$ from the center of gravity.

$$
F_p = \left(u_\mathrm{l} + u_\mathrm{r}\right) \begin{bmatrix}  \sin(\phi) \\
\cos(\phi)\end{bmatrix}
$$

- Gravity at the center of gravity

$$F_g = \begin{bmatrix}0 \\
-m g\end{bmatrix}$$
- Optional: aerodynamic drag force 

$$F_D(v, v_\mathrm{wind}) = ?$$

The forces of the propellors not only move the drone, but also create a moment 

$$
M_p = \left(u_\mathrm{l} - u_\mathrm{r}\right) d
$$

which rotates the drone around it's center of mass. The dynamics are then given by:

$$
\begin{aligned}
\begin{bmatrix}
\dot{p} \\
\dot{v} \\
\dot{\phi} \\
\ddot{\phi}
\end{bmatrix} = \dot{x} = f(x,u) =  \begin{bmatrix}
v \\
m^{-1}(F_p + F_g) \\
\dot{\phi} \\
I^{-1} M_p
\end{bmatrix}
\end{aligned}
$$

## Ideas for Projects
- (MEDIUM) Use an LQR/PID/H-$\infty$ controller to stabilize the drone.
	- (MEDIUM) Extend the drone model with a airdrag force, stabilize the drone against a strong wind gust.
	- (MEDIUM) USE an LQR controller to track a reference
- (MEDIUM) Plan an open-loop trajectory for a point-to-point motion by solving an OCP.
	- (HARD) Plan a time-optimal trajectory
- (MEDIUM) Implement a tracking MPC-Controller
	 - (HARD) Implement a collision-avoidance tracking MPC controller for two drones.

## Details

| State                                     | Symbol               | Unit          |
| ----------------------------------------- | -------------------- | ------------- |
| XZ - position of the drone                | $p \in \mathbb{R}^2$ | $\mathrm{m}$      |
| XZ - velocity of the drone                | $v \in \mathbb{R}^2$ | $\mathrm{m}\cdot \mathrm{s}^{-1}$     |
| orientation relative to the vertical axis | $\phi $              | $\mathrm{rad}$           |
| angular velocity                          | $\dot{\phi}$         | $\mathrm{rad}\cdot \mathrm{s}^{-1}$        |

| Parameter                   | Symbol | Value | Unit                      |
| --------------------------- | ------ | ----- | ------------------------- |
| distance to rotor           | $d$    | 0.20  | $\mathrm{m}$             |
| mass                        | $m$    | 0.50  | $\mathrm{kg}$             |
| rotational interia          | $I$    | 0.04  | $\mathrm{kg}\cdot\mathrm{m}^2$ |
| acceleration due to gravity | $g$    | 9.81  | $\mathrm{m}\cdot\mathrm{s}^{-2}$        |
