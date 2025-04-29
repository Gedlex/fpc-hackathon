# (Method) Model Predictive Control

Model predictive control is an advance control method, which find a closed-loop control $u(x)$ by solving a discrete optimal control problem in every iteration, given an (estimate of the) current state $\hat{x}$ of the system.

<img src="_misc/closedLoop.svg" width="600"/>

Since the discrete optimization problem predicts the future state evolution over a finite time horizon, its solution gives a control strategy that is both optimal in some objective and satisfies operational constraints of the system.

The problem solved is given by the following constrained nonlinear program:

$$
\begin{aligned}
\min_{x_0,u_0,x_1,u_1} &\sum_{k=0}^{N-1} l_k(x_k,u_k) + E(x_N) \\
\text{s.t.}\quad & 0 = x_0 - \hat{x} \\
&  0 = x_{k+1} - F(x_k, u_k), \quad &k=0,\dots,N-1 \\
&  0 \geq h(x_k, u_k), \quad &k=0,\dots,N-1 
\end{aligned}
$$

more details can be found in the [Open Loop Planning](../documentation/Method%20-%20Open%20Loop%20Planning.md) page.

After its solution, the first control $u_0$ is applied to the system. In the next time step, the optimization problem is solved again with the new state $\hat{x}$ of the system as initial state $x_0$.


## Further Literature:
- **Moritz Diehl and Sébastien Gros**, _Numerical Optimal Control_. Available online: [http://www.syscop.de/numericaloptimalcontrol](http://www.syscop.de/numericaloptimalcontrol).

- **J. B. Rawlings, D. Q. Mayne, and M. M. Diehl**, *Model Predictive Control: Theory, Computation, and Design*, 2nd edition, Nob Hill, 2017.

