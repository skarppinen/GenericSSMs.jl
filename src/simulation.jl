"""
   `Level{:state}, Level{:observation}, Level{(:state, :observation)}`

A struct tag (struct with no fields) representing the level (state level or observation level) \
at which simulation or prediction should be carried out.
This type is used in the functions related to simulating and predicting \
(functions `simulate!/simulate` and `predict!/predict`).

For example, using `Level{:state}` in the aforementioned functions yields predictions/simulations \
at the state level, and using `Level{:observation}` yields them at the observation level.
`Level{(:state, :observation)}` or `Level{(:observation, :state)}` yields predictions/simulations \
at both levels simultaneously. Note that `:observation` may be shortened to `:obs`. 
"""
struct Level{T} end
const StateLevel = Union{Level{:state}, Level{(:state,)}};
const ObsLevel = Union{Level{:obs}, Level{:observation}, Level{(:obs,)}, Level{(:observation,)}};
const StateAndObsLevel = Union{Level{(:state, :obs)}, Level{(:obs, :state)}, Level{(:state, :observation)},
                               Level{(:observation, :state)}};

"""
Extract type of simulation output of `model` if simulating at state level.
"""
@inline function simulation_type(model::GenericSSM, ::Type{<: StateLevel})
    particle_type(model);
end
"""
Extract type of simulation output of `model` if simulating at observation level.
"""
@inline function simulation_type(model::GenericSSM, ::Type{<: ObsLevel})
    observation_type(model);
end
"""
Extract type of simulation output of `model` if simulating at state and observation level.
"""
@inline function simulation_type(model::GenericSSM, ::Type{<: StateAndObsLevel})
    Tuple{particle_type(model), observation_type(model)};
end

function check_args_simulate(dest::AbstractVector, initial::Tuple{Integer, Any})
    n = length(dest);
    if n < 1
        msg = "length of `dest` must be >= 1."
        throw(ArgumentError(msg));
    end
    if initial[1] < 1
        msg = "`first[1]` should correspond to the first time index and thus should be >= 1.";
        throw(ArgumentError(msg));
    end
    nothing;
end

"""
   ```simulate(n::Integer, model::GenericSSM, ::Type{L <: Level}
               [, rng = Random.GLOBAL_RNG; initial::Tuple{Integer, P} = (1, m1(model, rng))])
   ```

Return a vector of length `n`, `dest`, by simulating at level `L` from model `model`. 
`L` specifies the level at which simulations should be carried out, and may be set to one of \
`Level{:state}, Level{:observation}, Level{(:state, :observation)}` (with possible abbreviations, see `?Level`) \
which results in simulations at state, observation or state and observation levels, respectively. 
In the final case, each simulated value corresponds to a tuple of the form `(p, o)`,
where `p` is a simulated particle value, and `o` a simulated observation simulated given `p`.

The value `initial = (k, x)` specifies that the time index associated with `dest[1]` should be `k`, and
that the state value associated with `dest[1]` is `x`. Indeed, if states are simulated `x` is placed to `dest[1]`. \
The default of `initial` corresponds to `(1, m1(model, rng))` meaning that simulation begins from the first time \
index. 
See `simulate!` for in place version.

REQUIRES:
* `m1`, `mk`, `gk` from the GenericSSMs interface (see `?GenericSSM`) depending on input `L`. 
"""
function simulate(n::Integer, model::GenericSSM, L::Type{<: Level}, rng = Random.GLOBAL_RNG;
        initial::Tuple{Integer, P} = (1, m1(model, rng))) where {P}
    if n < 1
        msg = "`n` should be >= 1.";
        throw(ArgumentError(msg));
    end
    dest = Vector{simulation_type(model, L)}(undef, n);
    simulate!(dest, model, L, rng; initial = initial);
    dest;
end

"""
   `simulate!(dest::AbstractVector{P}, model::GenericSSM, Level{:state}[, rng = Random.GLOBAL_RNG; initial::Tuple{Integer, P} = (1, m1(model, rng))])` 

Fill `dest` by simulating states from model `model`. 

The value `initial = (k, x)` specifies:
1. that `x` should be placed to `dest[1]`, 
2. that the remaining elements in indices `2:length(dest)` of `dest` should be simulated from `mj` where `j in (k + 1):(k + length(dest) - 1)`.
The default of `initial` corresponds to `(1, m1(model, rng))` meaning that the function returns a draw of state values at time indices `1:length(dest)`. 
See `simulate` for allocating version.

REQUIRES:
* `m1` and `mk` from the GenericSSMs interface (see `?GenericSSM`).
"""
function simulate!(dest::AbstractVector{P}, model::GenericSSM, ::Type{<: StateLevel}, 
        rng = Random.GLOBAL_RNG; initial::Tuple{Integer, P} = (1, m1(model, rng))) where P
    check_args_simulate(dest, initial); 
    n = length(dest);
    firstk, x = initial;
    @inbounds dest[1] = x;
    for k in (firstk + 1):(firstk + n - 1)
        i = k - firstk + 1; # NOTE: i runs through 2:n 
        @inbounds dest[i] = x = mk(model, x, k, rng);  
    end
    nothing;
end

"""
   `simulate!(dest::AbstractVector{Y}, model::GenericSSM, Level{:observation}[, rng = Random.GLOBAL_RNG; \
   initial::Tuple{Integer, P} = (1, m1(model, rng))])` 

Fill `dest` by simulating observations from model `model`. 

The value `initial = (k, x)` specifies:
1. that `x` is the state that is conditioned on when simulating the first observation to `dest[1]`, 
2. that the remaining observations in indices `2:length(dest)` of `dest` should be simulated conditional on states simulated from `mj` where `j in (k + 1):(k + length(dest) - 1)`.
The default of `initial` corresponds to `(1, m1(model, rng))` meaning that the function returns a draw of observations at time indices `1:length(dest)`.
See `simulate` for allocating version.

REQUIRES:
* `m1`, `mk` and `gk` from the GenericSSMs interface (see `?GenericSSM`). 
"""
function simulate!(dest::AbstractVector{Y}, model::GenericSSM, ::Type{<: ObsLevel}, rng = Random.GLOBAL_RNG;
        initial::Tuple{Integer, P} = (1, m1(model, rng))) where {Y, P}
    check_args_simulate(dest, initial);
    n = length(dest);
    firstk, x = initial;
    @inbounds dest[1] = gk(model, x, firstk, rng);
    for k in (firstk + 1):(firstk + n - 1)
        x = mk(model, x, k, rng);
        i = k - firstk + 1;
        @inbounds dest[i] = gk(model, x, k, rng); 
    end
    nothing;
end

"""
   `simulate!(dest::AbstractVector{Tuple{P, Y}}, model::GenericSSM, Level{(:state, :observation)}[, \
   rng = Random.GLOBAL_RNG; initial::Tuple{Integer, P} = (1, m1(model, rng))])` 

Fill `dest` by simulating states and observations from model `model`. 

The value `initial = (k, x)` specifies:
1. that `x` should be placed to `dest[1][1]`, 
2. that the remaining elements in indices `2:length(destx)` of `dest` should be simulated from `mj` and `gj` where `j in (k + 1):(k + length(dest) - 1)`.
The default of `initial` corresponds to `(1, m1(model, rng))` meaning that the function returns a draw of states and observations at time indices `1:length(dest)`. 
See `simulate` for allocating version.

REQUIRES:
* `m1`, `mk` and `gk` from the GenericSSMs interface (see `?GenericSSM`).
"""
function simulate!(dest::AbstractVector{Tuple{P, Y}}, model::GenericSSM, 
        ::Type{<: StateAndObsLevel}, rng = Random.GLOBAL_RNG; 
        initial::Tuple{Integer, P} = (1, m1(model, rng))) where {P, Y}
    check_args_simulate(dest, initial);
    n = length(dest);
    firstk, x = initial;
    @inbounds dest[1] = (x, gk(model, x, firstk, rng));
    for k in (firstk + 1):(firstk + n - 1)
        i = k - firstk + 1; # NOTE: i runs through 2:n 
        x = mk(model, x, k, rng);  
        @inbounds dest[i] = (x, gk(model, x, k, rng));
    end
    nothing;
end
