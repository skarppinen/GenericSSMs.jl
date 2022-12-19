# Defining SSMs using GenericSSMs.jl 

For a quick reference to the information provided here, type `?GenericSSM` in the Julia REPL after loading GenericSSMs.jl.

## Overview 

To use GenericSSMs.jl for the inference of an SSM, the user should define:

1. **A model struct representing the SSM.**
2. **Methods that implement the SSM.**

Steps 1 and 2 above are described below.

## Model struct and the abstract type `GenericSSM`

The model struct is a normal Julia `struct` that represents a user-defined SSM. 
The purpose of the model struct is to act as: 

1. a type for code selection using [multiple dispatch](https://docs.julialang.org/en/v1/manual/methods/), 
2. a container for data and parameters related to the SSM.

All model structs need to be defined as a subtype of `GenericSSM`, which
is an abstract type exported by GenericSSMs.jl.
Otherwise, the user is free to define the model struct as they see fit for their problem.

As an example, to define an SSM named `Model` that has floating point observations,
the following model struct could be used:
```
struct Model <: GenericSSM
   y::Vector{Float64}
   # .. arbitrary fields such as other data and parameters related to the SSM ..
end
```
Here, the fieldname `y` is an arbitrary choice. 
Naturally, it is also possible to use type parameters in the definition of the model struct.

There are two _optional_ methods that the user may wish to overload for their model struct:
```@docs
particle_type
```
```@docs
observation_type
```
As discussed above, both of these functions have fallbacks that work once the user defines 
some of the methods discussed in the next section.

The section [Examples](examples.md) contains examples of model struct definitions for more
concrete SSMs.

## Methods implementable for GenericSSMs 

After the model struct has been defined, the user should implement (at minimum a subset of, see [Method definitions required by use case](@ref)) methods that define 
* the components ``(M_{1:n}, G_{1:n})`` of the [Feynman-Kac model](ssms.md#Feynman-Kac-representation-of-a-state-space-model) of interest
* the components ``(m_{1:n}, g_{1:n})`` of the [underlying SSM](ssms.md#State-space-models).

The interface below gives the method signatures for all implementable methods in GenericSSMs.jl, with

* `Model <: GenericSSM` standing for the type of a user-defined model struct,
* `P` standing for any (particle) type,
* `F` standing for a floating point type, i.e `F <: AbstractFloat`,
* `Y` standing for any (observation) type, 
* the `rng` argument reserved for a generic random number generator.

!!! interface "All implementable methods in the GenericSSMs.jl interface for Model <: GenericSSM"
	1. `M1(model::Model, rng)::P` should return a simulated draw from ``M_1(\cdot)``.
	2. `logG1(model::Model, x::P)::F` should return ``\log(G_1(x)) \in \mathbb{R}``.
	3. `Mk(model::Model, x::P, k::Integer, rng)::P` should return a simulated draw from ``M_k(\cdot \mid x), k \geq 2``. 
	4. `logGk(model::Model, x::P, y::P, k::Integer)::F` should return ``\log(G_k(x, y)) \in \mathbb{R}, k \geq 2``.
	5. `logM1(model::Model, x::P)::F` should return ``\log(M_1(x)) \in \mathbb{R}``.
	6. `logMk(model::Model, y::P, x::P, k::Integer)::F` should return ``\log(M_k(y \mid x)) \in \mathbb{R}, k \geq 2``.
	7. `m1(model::Model, rng)::P` should return a simulated draw from ``m_1(\cdot)``.
	8. `mk(model::Model, x::P, k::Integer, rng)::P` should return a simulated draw from ``m_k(\cdot \mid x), k \geq 2``.
	9. `gk(model::Model, x::P, k::Integer, rng)::Y` should return a simulated draw from ``g_k(\cdot \mid x), k \geq 1``.

!!! note "In the above interface:"
	1. The first argument in each method signature is used to distinguish the methods of separate SSMs. 
	2. `Mk`, `logGk`, `logMk` and `mk` should be defined for ``k \geq 2``. The cases ``k = 1`` should be implemented by the separate functions `M1`, `logG1`, `logM1` and `m1`. This is necessary since the [mathematical function signatures](ssms.md) for the cases ``k \geq 2`` and ``k = 1`` differ. 
	3. The order of the arguments `x` and `y` in the signatures of `logGk` and `logMk` are reversed, again for consistency with the mathematical notation.
	4. The methods are _not_ exported by GenericSSMs.jl. When defining them, their names should be qualified by `GenericSSMs`.

See the [Examples](examples.md) section for examples of defining the above methods for concrete SSMs.

## Method definitions required by use case

The user need not define all methods in the interface of GenericSSMs.jl, but may instead define the part of the interface that is needed for their use case. 
The following table displays the required functions for each use case of GenericSSMs.jl. 

| Use case					          | M1 | logG1 | Mk | logGk | logM1 | logMk | m1 | mk | gk | 
|:--------------------------------|:--:|:-----:|:--:|:-----:|:-----:|:-----:|:--:|:--:|:--:| 
| Particle filtering   	          | x  |   x   | x  |  x    |       |       |    |    |    |
| CPF (with ancestor tracing)     | x  |   x   | x  |  x    |       |       |    |    |    |
| CPF (with backward sampling)    | x  |   x   | x  |  x    |       |   x   |    |    |    |
| Prediction at state level       | x  |   x   | x  |  x    |       |       |  x | x  |    |
| Prediction at observation level | x  |   x   | x  |  x    |       |       |  x | x  | x  |
| Simulation at state level       |    |       |    |       |       |       |  x | x  |    | 
| Simulation at observation level |    |       |    |       |       |       |  x | x  | x  | 

Note that `logM1` is currently not required for any use case, but is included in GenericSSMs.jl since it is required by some particle filtering algorithms (such as the particle Gibbs algorithm). 

!!! note
	A `MethodError` will be thrown if a particular method needed by a use case is not defined when the respective algorithm is invoked.

## Unicode method aliases 

GenericSSMs.jl also supports more aesthetically pleasing aliases that can be used instead of the above method names. 
The following table lists them: 

| Function | Alias      | 
|:---------|:-----------|
| M1       | M₁         |
| Mk       | Mₖ         |
| logG1    | logG₁      | 
| logGk    | logGₖ      |
| logM1    | logM₁      |
| logMk    | logMₖ      |
| m1       | m₁         |
| mk       | mₖ         |
| gk       | gₖ         |

Type `\_1[Tab]` and `\_k[Tab]` (where [Tab] is a Tab press) in the Julia REPL to write `₁` and `ₖ`, respectively.
