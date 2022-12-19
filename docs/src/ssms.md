# State-space models and Feynman-Kac representations

This section gives a (very brief) introduction to state-space models and their Feynman-Kac representations, and introduces some notation used in the documentation of GenericSSMs.jl.

## State-space models

State-space models (SSMs) are a broad class of statistical models for modelling multivariate time series data.
Suppose that we are interested in modelling a ``p``-dimensional time series of ``n`` observations, ``y_{1:n} = y_1, y_2, \ldots, y_n``. 
SSMs assume that ``y_{1:n}`` are generated conditional on a latent (unknown) state process, ``x_{1:n} = x_1, x_2, \ldots, x_n`` whose dynamics form a Markov process. 
A state-space model can thus be written in the following form:

```math
	\begin{aligned}
	\text{State process/equation:} \\
		x_1 &\sim m_1(\cdot) \\
		x_k &\sim m_k(\cdot \mid x_{k-1}) \ \text{ for } k \geq 2 \\
	\text{Observation process/equation:} \\
		y_k &\sim g_k(\cdot \mid x_k) \ \text{ for } k \geq 1 \\
	\end{aligned}
```
where:
* ``m_1`` is the distribution of the first latent state, ``x_1``.
* ``m_k(\cdot \mid x_{k-1})`` for ``k \geq 2`` are Markov transitions of the latent state, and
* ``g_k(y_k \mid x_k)`` for ``k \geq 1`` are densities of the observations ``y_k`` given the state ``x_k``.

In summary, the probability distributions ``m_k, k \geq 1`` constitute the dynamics of the latent state process ``x_{1:n}``, and ``g_k, k \geq 1`` describe how the (known) observations are generated conditional on the state process. State-space modelling means designing the state variables ``x_{1:n}``, their dynamics ``m_{1:n}`` and the observation densities ``g_{1:n}`` such that the model comprised of these provides a sensible statistical model for the observed data ``y_{1:n}``.

### Computational problem

Assume for simplicity that ``m_k, k \geq 1`` admit densities (although this assumption is necessary only for _some_ of the functionality of GenericSSMs).
Then, the joint density of the states ``x_{1:n}`` and observations ``y_{1:n}`` is given by:
```math
	\pi(x_{1:n}, y_{1:n}) = m_1(x_1)g_1(y_1 \mid x_1)\prod_{k = 2}^{n} m_k(x_k \mid x_{k-1}) g_k(y_k \mid x_k).
```

In the context of SSMs, the central computational problem is in the computation of various posterior distributions of the unknown latent states ``x_{1:n}``. 
Indeed, GenericSSMs provides functionality for simulating from the following posterior distributions: 

* ``p(x_{1:k} \mid y_{1:k}) \text{ for some } k \geq 1`` (filtering distributions) 

* ``p(x_{n + h} \mid y_{1:n})`` or ``p(y_{n + h} \mid y_{1:n})`` for some ``h > 0`` (predictive distributions)

* ``p(x_{1:n} \mid y_{1:n})`` (smoothing distribution)

Furthermore, [the particle filter](use-cases.md#Particle-filtering) also gives access to an estimate of the normalising constant, ``p(y_{1:n})``.

## Feynman-Kac representation of an SSM

In GenericSSMs.jl, SSMs are defined in terms of Feynman-Kac models [see Del Moral (2004), Chopin and Papaspiliopoulos (2020)] 
that are alternative representations for SSMs.
There are multiple possible Feynman-Kac models for a single (underlying) SSM. 
The rationale for using Feynman-Kac models is that they provide a convenient abstraction over SSMs that is very well suited for generic programming. 

A Feynman-Kac model expresses 
```math 
	\pi(x_{1:n} \mid y_{1:n}) \propto m_1(x_1)g_1(y_1 \mid x_1)\prod_{k = 2}^{n} m_k(x_k \mid x_{k-1}) g_k(y_k \mid x_k)
``` 
of the underlying SSM ``(m_{1:n}, g_{1:n})`` in terms of \`components\` ``(M_{1:n}, G_{1:n})``, such that it is assumed that
```math
	\pi(x_{1:n} \mid y_{1:n}) \propto M_1(x_1)G_1(x_1) \prod_{k = 2}^{n} M_k(x_k \mid x_{k-1}) G_k(x_{k-1}, x_k) \ \text{ for all } x_{1:n}.
```
The components ``(M_{1:n}, G_{1:n})`` consists of:
* ``M_1``, which is an (alternative) initial distribution for ``x_1`` 
* ``M_k`` for ``k \geq 2`` that are (alternative) Markov transitions of the state, and 
* ``G_k`` for ``k \geq 1`` that are \`potential functions\` taking values in the non-negative reals. 

In other words, the previous equation simply says that the joint distribution of a Feynman-Kac model must equal the joint posterior of the underlying SSM (up to a constant of proportionality). 

Note that in the above we have again assumed that ``M_k`` admit densities (although this is not required by all algorithms of GenericSSMs.jl, see [Method definitions required by use case](@ref)).  In general, ``M_k`` should however be simulatable, and ``G_k`` should be evaluatable pointwise. 

An example of a Feynman-Kac model ``(M_{1:n}, G_{1:n})`` for the SSM ``(m_{1:n}, g_{1:n})`` is given by the choice 
```math
	\begin{aligned}
		M_k &:= q_k \\
		G_k &:= \dfrac{g_k m_k}{q_k}, \\
	\end{aligned}
```
where ``q_k`` is a \`proposal distribution\` for the states. 
A special case of this Feynman-Kac model is the choice ``q_k = m_k``, which yields _the bootstrap filter_ of [Gordon, Salmond, Smith (1993)].

##### References

- _Del Moral, P. (2004) Feynman-Kac Formulae. Springer._
- _Chopin, N., and Papaspiliopoulos, O. (2020) An introduction to sequential Monte Carlo. Springer._
- _Gordon, N. J., Salmond, D. J., and Smith, A. F. (1993) Novel approach to nonlinear/non-Gaussian Bayesian state estimation. IEE Proceedings F (Radar and Signal Processing). 140(2):107-113._
