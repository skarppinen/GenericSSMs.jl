function check_storage_inputs(N::Integer, n::Integer)
	@assert N >= 2 "number of particles `N` should be >= 2";
	@assert n >= 1 "number of time steps `n` should be >= 1";
end

"""
A storage object used for standard particle filtering.
The object contains all memory needed for particle filtering. 

Type parameters:
* `P`: Particle type.
* `F`: Floating point type.

Fields:
* `X::Matrix{P}`, the `N` x `n` particle system of particles
* `W::Matrix{F}`, the `N` x `n` unnormalised logweight matrix,
* `A::Matrix{Int}`, the `N` x `n - 1` ancestor index matrix.

The additional field `_w` is used internally for normalised weights (do not modify).
The ancestor of the particle `X[i, k]` is `X[A[i, k - 1], k - 1]`.
"""
struct PFStorage{P, F <: AbstractFloat}
    X::Matrix{P}
    W::Matrix{F}
    A::Matrix{Int}

    _w::Vector{F}

   """
      `PFStorage(::Type{P}, N::Integer, n::Integer; F::Type{<: AbstractFloat} = Float64)`

   Construct a storage object for standard particle filtering with particle type `P`, 
   `N` particles and time series length `n`. `F` may be used to set the floating point type.

   ASSUMPTIONS:
   * `N >= 2`, `n >= 1` (checked, throws error)
   """
   function PFStorage(::Type{P}, N::Integer, n::Integer; F::Type{<: AbstractFloat} = Float64) where P
       check_storage_inputs(N, n);	
       X = Matrix{P}(undef, N, n);
       W = zeros(F, N, n);
       A = zeros(Int, N, n - 1);
       _w = zeros(F, N); 
       new{P, F}(X, W, A, _w);
   end
end
"""
   `PFStorage(model::GenericSSM, N::Integer, n::Integer; F::Type{<: AbstractFloat} = Float64)`

Construct a storage object for standard particle filtering model `model` with `N` particles
and time series length `n`. `F` may be used to set the floating point type.

ASSUMPTIONS:
* `N >= 2`, `n >= 1` (checked, throws error)
"""
function PFStorage(model::GenericSSM, N::Integer, n::Integer; F::Type{<: AbstractFloat} = Float64)
	P = particle_type(model);
	PFStorage(P, N, n, F = F);
end
"""
   `number_of_particles(storage::PFStorage)`

Return the number of particles that `storage` has storage for.
"""
@inline number_of_particles(storage::PFStorage) = size(storage.X, 1);

"""
   `length(storage::PFStorage)`

Return the time series length that `storage` has storage for.
"""
@inline length(storage::PFStorage) = size(storage.X, 2);

"""
A storage object used for conditional particle filtering.
The object contains all memory needed for conditional particle filtering.

Type parameters:
* `P`: Particle type.
* `F`: Floating point type.

Fields:
* `pfst::PFStorage{P, F}`, the storage used for standard particle filtering (see `?PFStorage`),
* `ref::Vector{Int}`, a vector of length `n` that records which \
rows (elements) in each column of `pfst.X` currently hold the reference trajectory.
"""
struct CPFStorage{P, F <: AbstractFloat}
	pfst::PFStorage{P, F}
	ref::Vector{Int}
   """
      `CPFStorage(pfst::PFStorage)`

   Construct a storage object for conditional particle filtering using a 
   `PFStorage` object.
   """
   function CPFStorage(pfst::PFStorage{P, F}) where {P, F}
      n = length(pfst);
		ref = zeros(Int, n);
      new{P, F}(pfst, ref);
   end
end

"""
      `CPFStorage(::Type{P}, N::Integer, n::Integer; F::Type{<: AbstractFloat} = Float64)`

   Construct a storage object for conditional particle filtering with particle type `P`, 
   `N` particles and time series length `n`. `F` may be used to set the floating point type.

   ASSUMPTIONS:
   * `N >= 2`, `n >= 1` (checked, throws error)
   """
function CPFStorage(::Type{P}, N::Integer, n::Integer; F::Type{<: AbstractFloat} = Float64) where P
    pfst = PFStorage(P, N, n, F = F);
    CPFStorage(pfst); 
end
"""
   `CPFStorage(model::GenericSSM, N::Integer, n::Integer; F::Type{<: AbstractFloat} = Float64)`

Construct a storage object for conditional particle filtering model `model` with `N` particles
and time series length `n`. `F` may be used to set the floating point type.

ASSUMPTIONS:
* `N >= 2`, `n >= 1` (checked, throws error)
"""
function CPFStorage(model::GenericSSM, N::Integer, n::Integer; F::Type{<: AbstractFloat} = Float64) 
    P = particle_type(model);
    pfst = PFStorage(P, N, n, F = F);
    CPFStorage(pfst); 
end

"""
   `number_of_particles(storage::CPFStorage)`

Return the number of particles that `storage` has storage for.
"""
@inline number_of_particles(storage::CPFStorage) = number_of_particles(storage.pfst);


"""
   `length(storage::CPFStorage)`

Return the time series length that `storage` has storage for.
"""
@inline Base.length(storage::CPFStorage) = length(storage.pfst); 

"""
   `size(storage::Union{PFStorage, CPFStorage})`

Return a tuple `(N, n)`, where `N` is the number of particles and `n` is the 
time series length.
"""
@inline function Base.size(storage::Union{PFStorage, CPFStorage}) 
    (number_of_particles(storage), length(storage));
end

function check_args_predict_storage(X::Matrix, x::Vector, w::Vector, n::Integer) 
    sx = size(X);
    if sx[1] <= 0
        msg = "`nahead = size(X, 1)` should be > 0.";
        throw(ArgumentError(msg));
    end
    if sx[2] <= 0
        msg = "`nsim = size(X, 2)` should be > 0.";
        throw(ArgumentError(msg));
    end
    if n <= 0
        msg = "`n` must be > 0.";
        throw(ArgumentError(msg));
    end
    if length(x) != length(w)
        msg = "the lengths of `x` and `w` must match.";
        throw(ArgumentError(msg));
    end
    if length(x) <= 0
        msg = "the lengths of `x` and `w` must be >= 1";
        throw(ArgumentError(msg));
    end
end

"""
   `PredictStorage{S, P, L, F}`

A storage object needed for performing predictions at state, observation or state and observation level. 

Type parameters:
* `S` is the type of predicted values (either the particle type, observation type or tuple consisting of the particle type \
and observation type).
* `P` is the particle type.
* `L` is the level associated with the prediction storage.
* `F` is a floating point type.

Fields:
* `X::Matrix{S}`: An `nahead x nsim` matrix for storing `nsim` simulated \
trajectories `nahead` steps ahead from `n`.
* `x::Vector{P}`: The initial particles at time `n`.
* `w::Vector{F}`: The normalised weights of the initial particles `x`.
* `n::Int`: The time index associated with the weighted particles `(x, w)`.
* `_u::Vector{F}, _ind::Vector{Int}`: Temporaries used in predicting (best left untouched).
"""
struct PredictStorage{S, P, L <: Level, F <: AbstractFloat}
    X::Matrix{S}
    x::Vector{P}
    w::Vector{F}
    n::Int

    _u::Vector{F}
    _ind::Vector{Int}
    function PredictStorage(X::Matrix{S}, x::Vector{P}, w::Vector{F}, n::Integer, 
            L::Type{<: Level}) where {S, P, F <: AbstractFloat}
        check_args_predict_storage(X, x, w, n);
        nahead, nsim = size(X);
        _u = Vector{F}(undef, nsim);
        _ind = Vector{Int}(undef, nsim);
        new{S, P, L, F}(X, x, w, n, _u, _ind);
    end
end
function PredictStorage(st::PFStorage, model::GenericSSM, L::Type{<: Level}, nahead::Integer, nsim::Integer,  
        n::Integer = length(st)) 
    if n > length(st) 
        msg = "`n` should be less than or equal to the length of the storage `st`.";
        throw(ArgumentError(msg));
    end
    X = Matrix{simulation_type(model, L)}(undef, nahead, nsim); 
    w = normalise_logweights(view(st.W, :, n));
    x = st.X[:, n];
    PredictStorage(X, x, w, n, L);
end
"""
   ```PredictStorage(st::PFStorage, model::GenericSSM, L::Type{<: Level};
                     nahead::Integer, nsim::Integer, n::Integer = length(st))
   ```

Construct a PredictStorage object from `st` and `model` at level `L`. Specifically \
reserve memory for predicting `nahead` steps ahead using `nsim` simulations and starting at timepoint `n` \
of the storage `st`.

ASSUMPTIONS: 
* `n` is less than or equal to the length of `st` (checked, throws error)
* `pf_forward_pass!` has been run for `st` beforehand.
"""
function PredictStorage(st::PFStorage, model::GenericSSM, L::Type{<: Level};
        nahead::Integer, nsim::Integer, n::Integer = length(st)) 
    PredictStorage(st, model, L, nahead, nsim, n);
end
Base.size(st::PredictStorage) = size(st.X); 

"""
   `level(st::PredictStorage{P, S, L}) = L`

Returns the level associated with `st`.
"""
@inline function level(st::PredictStorage{P, S, L}) where {P, S, L} 
    L;
end
