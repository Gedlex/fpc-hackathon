# (Model) 2D-XY Bicycle Control
*Future PhD workshop, Hannover, April 2025, Jakob Harzer, Yunfan Gao, Moritz Diehl*

<img src="_misc/BicycleXYfigure.png" width="500"/>

## Dynamics:

$$
\begin{aligned}
x = \begin{bmatrix}
p_x \\
p_y \\
\theta
\end{bmatrix} \in \mathrm{R}^3 && u = \begin{bmatrix}
\delta \\
V
\end{bmatrix} \in \mathrm{R}^2
\end{aligned}
$$

with position $p_x$, $p_y$, and heading angle $\theta$ relative to the $x$-axis. The vehicle is controlled with a specific steering angle $\delta$ and a fixed velocity $V$.

$$
\begin{aligned}
\begin{bmatrix}
\dot{p}_ x \\
\dot{p}_ y \\
\dot{\theta}
\end{bmatrix} = \dot{x} = f(x,u) =  \begin{bmatrix}
V \cos(\theta + \beta) \\
V \sin(\theta + \beta) \\
\frac{V}{l_\mathrm{r}}\sin(\beta)
\end{bmatrix}
\end{aligned}
$$

The *side-slip angle* $\beta$ is given as

$$
\beta = \arctan\left(\frac{l_\mathrm{r} \tan(\delta)}{l_\mathrm{r} + l_\mathrm{f}}\right)
$$

and depends on the distances $l_\mathrm{r},l_\mathrm{f}$, of the center of mass to the front and rear wheels, in the plot we have $L = l_\mathrm{r} + l_\mathrm{f}$.

## Ideas for Projects
- (MEDIUM) collision avoidance (Open Loop Planning)
- (MEDIUM) car parking problem (Open Loop Planning)
- (HARD) collision avoidance (MPC)


## Details

| State                                     | Symbol               | Unit  |
| ----------------------------------------- | -------------------- | ----- |
| p                                         | $p \in \mathbb{R}^2$ | $\mathrm{m}$      |
| orientation relative to the vertical axis | $\phi \in \mathbb{R}$               | $\mathrm{rad}$    |

| Parameter                              | Symbol         | Value | Unit          |
| -------------------------------------- | -------------- | ----- | ------------- |
| distance center of mass to front wheel | $l_\mathrm{f}$ | 0.5   | $\mathrm{m}$              |
| distance center of mass to front wheel | $l_\mathrm{r}$ | 0.5   | $\mathrm{m}$ |
