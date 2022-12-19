module GenericSSMs
import Base: length
using Random
using Statistics: mean
using LinearAlgebra: normalize!
using Resamplings: StratifiedResampling, _ascending_inv_cdf_lookup!, generate_ordered_uniforms!, resample!,
                   conditional_resample!

@doc raw"""
An abstract type representing a generic state space model (GenericSSM) with \
a particular Feynman-Kac representation.

Here, a quick reference for model struct definition and interface function signatures are given.
See online documentation for further explanations.

## Example model struct definition:
```
struct Model <: GenericSSM # Remember to define as subtype of `GenericSSM`
   # ..arbitrary fields..
end
```
The fields of the model struct may contain any data, such as model parameters, \
observations, scratch memory etc. 

## Interface function signatures

Below:
* `Model` corresponds to the model struct defined by the user. 
* `P to any particle type (defined by the user or wherever).
* `Y` corresponds to any observation type (defined by the user or wherever).
* `F` is a subtype of `AbstractFloat`.

* `M1(model::Model, rng)::P` should simulate from M_1.
* `Mk(model::Model, prev::P, k::Integer, rng)::P` should simulate from \
``M_k(. \mid prev)`` for k >= 2.
* `logG1(model::Model, cur::P)::F` should evaluate log(G_1(cur)).
* `logGk(model::Model, prev::P, cur::P, k::Integer)::F` \
should evaluate log(G_k(prev, cur)) for k >= 2.
* `logMk(model::Model, cur::P, prev::P, k::Integer)::F` \
should return log(M_k(cur | prev)) for k >= 2.
* `logM1(model::Model, x::P)::F` should return log(M_1(x)).
* `m1(model::Model, rng)::P` should simulate from m1.
* `mk(model::Model, prev::P, k::Integer, rng)::P` \
should simulate from m_k(. | prev) k >= 2.
* `gk(model::Model, cur::P, k::Integer, rng)::Y` \
should simulate from g_k(. | prev) k >= 1.

Note that it is also possible to use the Unicode aliases M₁, Mₖ, logG₁, logGₖ, \
logM₁, logMₖ, m₁, mₖ, gₖ for the above methods. 

## Further optional interface function signatures (these have fallbacks implemented using the above):
* `particle_type(model::Model)` should return the particle type \
that is used with `model`.
* `observation_type(model::Model)` should return the observation type \
that is used with `model`.
"""
abstract type GenericSSM end

export GenericSSM, PFStorage, CPFStorage, pf_forward_pass!, cpf_forward_pass!,
       traceback!, get_reference!, initialise_reference!, particle_type, observation_type,
       AncestorTracing, BackwardSampling, use_resampling_backend,
       simulate!, simulate, predict!, predict, Level, level, PredictStorage

include("fallbacks.jl");
include("simulation.jl");
include("api.jl");
include("storage.jl");
include("prediction.jl");
include("forward-pass.jl");
include("traceback.jl");
include("helpers.jl");

end
