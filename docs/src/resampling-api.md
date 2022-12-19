# Resampling API

This section describes the resampling API of GenericSSMs.jl and is intended for users who wish to use their own resampling algorithm
with GenericSSMs.jl.
Otherwise, the resampling algorithms of [Resamplings.jl](https://github.com/skarppinen/Resamplings.jl) implement the required 
functionality and can be directly used with GenericSSMs.jl.

For particle filtering (`pf_forward_pass!`), the `resampling` object should implement:
```
resample!(resampling, ind::AbstractVector{<: Integer}, 
          w::AbstractVector{<: AbstractFloat}, rng)::Nothing
```
`resample!` should draw ancestor indices to `ind` given normalised weights `w`. 
Note that `length(ind) == length(w)`` should hold. The last argument is preserved for a random number generator.

For conditional particle filtering (`cpf_forward_pass!`), the `resampling` object should implement: 
```
conditional_resample!(resampling, ind::AbstractVector{<: Integer}, 
                      w::AbstractVector{<: AbstractFloat}, 
                      k::Integer, i:::Integer, rng)::Nothing
```
`conditional_resample!` should draw ancestor indices to `ind` given normalised weights `w` and the condition that `ind[k] = i`.
Again, `length(ind) == length(w)` should hold, and additionally, `w[i]` should be strictly positive.
The last argument is preserved for a random number generator.

In general the `resampling` object can be of any type (and naturally contain arbitrary fields), as long as it satisfies the above API.
It may be useful to look at the documentation and source code of [Resamplings.jl](https://github.com/skarppinen/Resamplings.jl) for examples.
