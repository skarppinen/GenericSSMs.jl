

function predict(st::PFStorage, newmodel::GenericSSM, L::Type{<: Level}, nahead::Integer, nsim::Integer,  
                 rng = Random.GLOBAL_RNG)
    predst = PredictStorage(st, newmodel, L, nahead, nsim, length(st));
    predict!(predst, newmodel, rng);
    predst;
end
"""
   `predict(st::PFStorage, newmodel::GenericSSM[, L = Level{:state}, rng = Random.GLOBAL_RNG]; \
   nahead::Integer, nsim::Integer)::PredictStorage`

Predict `nahead` steps ahead using `nsim` simulations at level `L` (see `?Level`).
`st` should be a `PFStorage` object that contains results of particle filtering `n` time steps (see below).
`newmodel` should be a model struct that contains the data (if any) necessary in making \
the future predictions. 
In particular, if `newmodel` contains data indexed by time in the definition of `mk`, this function \
requires that said data must be indexable at the future indices `n + 1, n + 2, ..., n + nahead`,
where `n = length(st)`, that is, the number of time points that have been filtered.
(otherwise a bounds error is thrown if bounds checking is not disabled in the definition of `mk` by the user)
To this end, the Julia package `OffsetArrays.jl` offers convenient ways of defining arrays whose indexing \
may be offset by `n` (upon construction of `newmodel`).

The return value is an object of type `PredictStorage` (see `?PredictStorage`), containing the simulations.
See `predict!` for non-allocating version of this function.

REQUIRES:
* `mk` and `gk` from the GenericSSMs interface (see `?GenericSSM`), depending on the value of `L`.

ASSUMPTIONS:
* `pf_forward_pass!` has been run with `st` as the first argument. Results WILL BE WRONG \
if this is not the case.
"""
function predict(st::PFStorage, newmodel::GenericSSM, L::Type{<: Level} = Level{:state}, rng = Random.GLOBAL_RNG;
        nahead::Integer, nsim::Integer)::PredictStorage
   predict(st, newmodel, L, nahead, nsim, rng);
end

"""
   `predict!(st::PredictStorage, newmodel::GenericSSM[, rng = Random.GLOBAL_RNG])`

Predict in place to storage `st` using model `newmodel`. The prediction is done at the level \
at which `st` was constructed. (see `?PredictStorage`)

`newmodel` should be a model struct that contains the data (if any) necessary in making \
the future predictions. 
In particular, if `newmodel` contains data indexed by time in the definition of `mk`, this function \
requires that said data must be indexable at the future indices `n + 1, n + 2, ..., n + size(st, 1)`
at which predictions are requested. 
(otherwise a bounds error is thrown if bounds checking is not disabled in the definition of `mk` by the user)

To this end, the Julia package `OffsetArrays.jl` offers convenient ways of defining arrays whose indexing \
may be offset by `n` (upon construction of `newmodel`).
See `predict` for allocating versions of this function.

REQUIRES: `mk` and `gk` from the GenericSSMs interface (see `?GenericSSM`), depending on the value of `L`.
"""
function predict!(st::PredictStorage, newmodel::GenericSSM, rng = Random.GLOBAL_RNG) 
    nahead, nsim = size(st);
    n = st.n; 
    lev = level(st);

    # Do stratified resampling.
    generate_ordered_uniforms!(StratifiedResampling, st._u, rng);
    _ascending_inv_cdf_lookup!(st._ind, st.w, st._u);
      
    # Predict.
    for j in Base.OneTo(nsim);
        x = view(st.X, :, j);
        @inbounds initial = (n + 1, mk(newmodel, st.x[st._ind[j]], n + 1, rng));
        simulate!(x, newmodel, lev, rng; initial = initial);
    end
    nothing;
end

