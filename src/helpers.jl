"""
   `normalise_logweights!(log_weights::AbstractVector{<: Real})`

Normalise a vector of unnormalised weight logarithms, `log_weights`, in place.
After normalisation, the weights are normalised and in the linear scale.
Additionally, the logarithm of the linear scale mean unnormalised weight is returned.
Note that the maximum of `log_weights` can not be `-Inf`, `Inf` or `NaN` since in \
these cases it is impossible to normalise. (an error is thrown in these cases)
See `normalise_logweights` for allocating version.
"""
function normalise_logweights!(log_weights::AbstractVector{<: Real})
  m = maximum(log_weights);
  if isinf(m) || isnan(m) 
      msg = string("maximum of input logweights is ", string(m), ", impossible to normalise");
      throw(ArgumentError(msg)) 
  end 
  log_weights .= exp.(log_weights .- m);
  log_mean_weight = m + log(mean(log_weights));
  normalize!(log_weights, 1);
  log_mean_weight;
end

"""
   `normalise_logweights(log_weights::AbstractVector{<: Real})`

Return a normalised vector of weights given a vector of unnormalised weight logarithms, `log_weights`.
Note that the maximum of `log_weights` can not be `-Inf`, `Inf` or `NaN` since in \
these cases it is impossible to normalise. (an error is thrown in these cases)
See `normalise_logweights!` for in place version.
"""
@inline function normalise_logweights(log_weights::AbstractVector{<: Real})
   x = copy(log_weights);
   normalise_logweights!(x);
   x;
end

"""
Sample one index from 1:length(`w`) proportional on the weights in `w`.

ASSUMPTIONS:
* `w` is normalised to 1.0. (RESULTS WRONG IF NOT THE CASE)
* `w` is non-empty. (RESULTS WRONG IF NOT THE CASE)
"""
@inline function wsample_one(w::AbstractVector{<: AbstractFloat},
                             rng::AbstractRNG = Random.GLOBAL_RNG)
  u = rand(rng);
  s = zero(eltype(w));
  for i in eachindex(w)
    s += @inbounds w[i];
    if u <= s
      return i;
    end
  end
  length(w);
end
