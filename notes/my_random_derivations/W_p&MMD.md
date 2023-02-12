Wasserstein and MMD are two well-known metrics over distributions. [https://arxiv.org/pdf/2112.00423.pdf](https://arxiv.org/pdf/2112.00423.pdf) studies necessary and
sufficient conditions for $W_1 \geq MMD $.

Following is a short proof that presents conditions on $\lambda \geq 1$ such that $W_1 \geq \lambda MMD ~ \forall s, t\in \Delta$ over a bounded input domain. Here, we assume
a normalized characteristic kernel used in MMD, hence, with the associated RKHS $\mathcal{H_k}$, we have that the max norm of  
$f_H\in \mathcal{H_k}$, denoted by $z$ is less than or equal to 1.

$$W_1(s, t)-\lambda MMD(s, t) \geq W_1(s, t)-2\lambda ||f_H||_\infty TV(s, t) \geq W_1(s, t)-d_{min} TV(s, t) \geq 0$$ 

For the second-last inequality, we need $1 \leq \lambda \leq \frac{d_{min}}{2z}$
where
 $d_{min}=\min_{x\neq x'} \rho c(x, x')$. Here $c$ is a ground metric
and $\rho\in \mathbb{R}^{++}$ is chosen such that $d_{min}>2$. Note that $\rho c$ is our new cost metric.

The last inequality comes from the relation shown in equation (7) of [On Choosing and Bounding Probability Metrics](https://arxiv.org/pdf/math/0209021.pdf). Note that this holds true only for a bounded metric space.

As remarked by Villani's classical OT book, $W_p \geq W_1$ for $p\geq 1$ which implies that the above inequality holds for $W_p$ also.

*Potential applications: in some MMD-regularized OT loss...*
