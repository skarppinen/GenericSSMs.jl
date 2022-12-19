
@doc raw"""
GenericSSMs.jl interface function.

`M1(model::Model, rng)::P` should return a simulated draw from ``M₁(·)``,
where `P` is the particle type.
This function may be aliased with `M₁`.
"""
function M1 end
const M₁ = M1;

"""
GenericSSMs.jl interface function.

`Mk(model::Model, x::P, k::Integer, rng)::P` should return a simulated draw from ``Mₖ(· ∣ x), k ≥ 2``,
where `P` is the particle type.
This function may be aliased with `Mₖ`.
"""
function Mk end
const Mₖ = Mk;

"""
GenericSSMs.jl interface function.

`logG1(model::Model, x::P)::F` should return ``log(G₁(x)) ∈ ℝ``,
where `P` is the particle type and `F` is a floating point type.
This function may be aliased with `logG₁`.
"""
function logG1 end
const logG₁ = logG1;

"""
GenericSSMs.jl interface function.

`logGk(model::Model, x::P, y::P, k::Integer)::F` should return ``log(⁡Gₖ(x, y)) ∈ R, k ≥ 2``,
where `P` is the particle type and `F` is a floating point type.
This function may be aliased with `logGₖ`.
"""
function logGk end
const logGₖ = logGk;

"""
GenericSSMs.jl interface function.

`logM1(model::Model, x::P)::F` should return ``log(M₁(x)) ∈ R``,
where `P` is the particle type and `F` is a floating point type.
This function may be aliased with `logM₁`.
"""
function logM1 end
const logM₁ = logM1;

"""
GenericSSMs.jl interface function.

`logMk(model::Model, y::P, x::P, k::Integer)::F` should return ``log(⁡Mₖ(y ∣x)) ∈ R, k ≥ 2``,
where `P` is the particle type and `F` is a floating point type.
This function may be aliased with `logMₖ`.
"""
function logMk end
const logMₖ = logMk;

"""
GenericSSMs.jl interface function.

`m1(model::Model, rng)::P` should return a simulated draw from ``m₁(·)``,
where `P` is the particle type.
This function may be aliased with `m₁`.
"""
function m1 end
const m₁ = m1;

"""
GenericSSMs.jl interface function.

`mk(model::Model, x::P, k::Integer, rng)::P` should return a simulated draw from ``mₖ(⋅ ∣ x), k ≥ 2``,
where `P` is the particle type.
This function may be aliased with `mₖ`.
"""
function mk end
const mₖ = mk;

"""
GenericSSMs.jl interface function.

`gk(model::Model, x::P, k::Integer, rng)::Y` should return a simulated draw from ``gₖ(· | x), k ≥ 1``,
where `P` is the particle type and `Y` is the observation type. 
This function may be aliased with `gₖ`.
"""
function gk end
const gₖ = gk;


