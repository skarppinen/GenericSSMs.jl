# Using GenericSSMs.jl

This section discusses how to use GenericSSMs.jl for SSMs defined as discussed in Section [Defining SSMs using GenericSSMs.jl](interface.md).

Most of the methods exported by GenericSSMs.jl do not (heap) allocate memory, but instead operate on `storage` objects, such that a typical method call (in this case to a method named `operation`) takes the form: 
```
operation!(storage, ... additional arguments ...)
```
The object `storage` is simply an object that collects all pre-allocated memory necessary for making a particular computation (such as particle filtering or predicting).
There are also \`allocating versions\` (lacking the exclamation mark) of some of the functionality, but the non-allocating versions should be used for performance especially in situations where a particular method needs to called multiple times (see [Julia performance tips: pre-allocating outputs](https://docs.julialang.org/en/v1/manual/performance-tips/#Pre-allocating-outputs)).

There are three storage objects used in GenericSSMs.jl:

* `PFStorage` (for particle filtering)
* `CPFStorage` (for conditional particle filtering)
* `PredictStorage` (for prediction)

These are discussed below with their respective use cases. In what follows, we will in general use `N` to refer to the number of particles, and `n` to the number of time points (or length of the observed time series).

## Particle filtering

The particle filter implementation of GenericSSMs.jl can be invoked by calling `pf_forward_pass!`: 
```@docs
pf_forward_pass!(::PFStorage, ::GenericSSM, ::Any, ::Any)
```
`pf_forward_pass!` populates its first argument, a `PFStorage` object, which has the following structure:
```@docs
PFStorage
```
The easiest way to construct a `PFStorage` object is via the constructor:
```@docs
PFStorage(::GenericSSM, ::Integer, ::Integer; ::Type{<: AbstractFloat})
```

The other arguments to `pf_forward_pass!` above are
* the SSM `model` (expected to be defined as in [Defining SSMs using GenericSSMs.jl](interface.md))
* `resampling` object, that implements `resample!` from the [resampling API](resampling-api.md) of GenericSSMs.jl 

For the latter, it is possible to use any resampling provided by [Resamplings.jl](https://github.com/skarppinen/Resamplings.jl) (see Section [Examples](examples.md) for examples of this).

Finally, for convenience, `pf_forward_pass!` can also be called with a `CPFStorage` storage object (discussed in [Conditional particle filtering](@ref)), as follows:
```@docs
pf_forward_pass!(::CPFStorage, ::GenericSSM, ::Any, ::Any)
```

## Conditional particle filtering

The conditional particle filter (CPF) is otherwise similar to the (standard) particle filter, but features a \`reference trajectory\` that is never mutated as the algorithm proceeds. Due to this, the CPF requires its own storage object, `CPFStorage`:
```@docs
CPFStorage
```

The construction of a `CPFStorage` object is similar to that of `PFStorage`:
```@docs
CPFStorage(::GenericSSM, ::Integer, ::Integer; ::Type{<: AbstractFloat})
```

The main function for running the (forward) conditional particle filter is called `cpf_forward_pass!`:
```@docs
cpf_forward_pass!
```
Here, the `resampling` object must implement `conditional_resample!` from the [resampling API](resampling-api.md) of GenericSSMs.jl.
Again, any _conditional_ resampling from [Resamplings.jl](https://github.com/skarppinen/Resamplings.jl) may be used.

Note that care must be taken to ensure that the reference trajectory has been initialised to a sensible value before calling `cpf_forward_pass!`.
For the first run of `cpf_forward_pass!`, a straightforward way to do this is to call `initialise_reference!` which uses the standard particle filter to initialise the reference:
```@docs
initialise_reference!
```

To do the CPF backward pass, the function `traceback!` can be used, with either `AncestorTracing` or `BackwardSampling`. 
This effectively records a new reference trajectory to `CPFStorage`. 
```@docs
traceback!(::CPFStorage, ::Any, ::Type{AncestorTracing})
traceback!(::CPFStorage, ::Type{AncestorTracing})
```
```@docs
traceback!(::CPFStorage, ::GenericSSM, ::Type{BackwardSampling})
```

The function `get_reference!` may be used to read the concrete value of the reference trajectory to a vector of appropriate type and length:
```@docs
get_reference!
```

## Unconditional simulation and prediction from SSMs

GenericSSMs.jl provides functionality for unconditional simulation and prediction from generic SSMs at state and/or observation levels.
The level at which simulation or prediction should be performed, is indicated by using a parametric type `Level`:
```@docs
Level
```

### Unconditional simulation

Unconditional simulation means simulating a continuous slice of states ``x_{l}, x_{l+1}, \ldots, x_u`` from the prior of ``x_{l:u}``,  where ``l < u`` and ``l \geq 1`` such that the initial state ``x_l`` is given as an argument. Similarly, a continuous slice of observations ``y_{l:u}`` may be drawn. (see below)
This functionality is achieved by the following allocating and nonallocating versions of `simulate`.
By default, ``l = 1`` and ``x_1`` is first drawn from ``m_1`` and then conditioned on when simulating ``x_2, \ldots, x_u``, but this may be changed with the argument `initial`:  

```@docs
simulate
```
```@docs
simulate!
```

### Prediction 

Prediction means simulating from the posterior distribution of future states ``\pi(x_{n+1:n+h} \mid y_{1:n})`` or observations ``\pi(y_{n+1:n+h} \mid y_{1:n})`` for some prediction horizon ``h \geq 1``.
Prediction requires that the particle filter has been run. 
The current implementation of prediction draws particles from the final particle approximation returned by the particle filter using stratified resampling and then simulates future state and/or observation trajectories given the chosen particles.

The allocating version of `predict` takes the form: 
```@docs
predict
```

Prediction may also be carried out in a non-allocating fashion, by first explicitly constructing a `PredictStorage` object: 
```@docs
PredictStorage
```
The easiest way to do this is via the constructor:
```@docs
PredictStorage(::PFStorage, ::GenericSSM, ::Type{<:Level}; ::Integer, ::Integer, ::Integer)  
```
Note that changing the argument `n` above to a value less than `length(st)` implicitly assumes that 
```math
	\pi(x_{1:n} \mid y_{1:n}) \propto M_1(x_1)G_1(x_1) \prod_{k = 2}^{n} M_k(x_k \mid x_{k-1}) G_k(x_{k-1}, x_k) \ \text{ for all } x_{1:n}.
```
holds also for this new value of `n` (and not only for `n = length(st)` as discussed in [Feynman-Kac representation of an SSM](ssms.md#Feynman-Kac-representation-of-an-SSM)). 

Then, `predict!` can be called with the storage object as the first argument:
```@docs
predict!
```
The prediction results can then be obtained from the `PredictStorage` object written to.
