"""
An abstract type representing a tracebacking method. 
"""
abstract type Traceback end

"""
   `AncestorTracing <: Traceback`

The original ancestor tracing variant of the conditional particle filter \
of Andrieu, Doucet, Holenstein (2010).
"""
struct AncestorTracing <: Traceback end
"""
   `BackwardSampling <: Traceback`

The backward sampling variant (Whiteley, 2010) of the conditional particle filter \
of Andrieu, Doucet, Holenstein (2010).
"""
struct BackwardSampling <: Traceback end

"""
   `traceback!(st::CPFStorage, model::GenericSSM, AncestorTracing[, rng = Random.GLOBAL_RNG])`

Populate the reference indices `st.ref` using ancestor tracing. 
To obtain the concrete reference trajectory values, use `get_reference!`.

ASSUMPTIONS:
* the forward pass (`pf_forward_pass!` or `cpf_forward_pass!`) has \
been run before invoking this function and the contents of `st` \
have not changed after that. The results from this function WILL BE WRONG, if this is not the case. 
"""
@inline function traceback!(st::CPFStorage, model, ::Type{AncestorTracing},
                            rng::AbstractRNG = Random.GLOBAL_RNG) 
    ref = st.ref; A = st.pfst.A; _w = st.pfst._w;
    N, n = size(st); 
    @inbounds ref[n] = a = wsample_one(_w, rng);
    for k in (n - 1):-1:1
        @inbounds ref[k] = a = A[a, k];
    end
    nothing;
end

"""
   `traceback!(st::CPFStorage, AncestorTracing[, rng = Random.GLOBAL_RNG])`

Convenience method for traceback using ancestor tracing (model not needed by ancestor tracing).

ASSUMPTIONS:
* the forward pass (`pf_forward_pass!` or `cpf_forward_pass!`) has \
been run before invoking this function and the contents of `st` \
have not changed after that. The results from this function WILL BE WRONG, if this is not the case. 
"""
@inline function traceback!(st::CPFStorage, ::Type{AncestorTracing},
                            rng = Random.GLOBAL_RNG)
   traceback!(st, nothing, AncestorTracing, rng);    
end

"""
   `traceback!(st::CPFStorage, model::GenericSSM, BackwardSampling[, rng = Random.GLOBAL_RNG])`

Populate reference indices `st.ref` using backward sampling \
after running the forward pass, that is, `pf_forward_pass!` or `cpf_forward_pass!`. \
To obtain the concrete reference trajectory values, see `get_reference!`.

ASSUMPTIONS:
* the forward pass (`pf_forward_pass!` or `cpf_forward_pass!`) has \
been run before invoking this function and the contents of `st` \
have not changed after that. The results from this function WILL BE WRONG, if this is not the case. 
"""
@inline function traceback!(st::CPFStorage, 
                            model::GenericSSM, 
                            ::Type{BackwardSampling},
                            rng = Random.GLOBAL_RNG)
    X = st.pfst.X; W = st.pfst.W; ref = st.ref; 
    _w = st.pfst._w;
    N, n = size(st);

    # Sample indices and save to `ref`.
    @inbounds ref[n] = ai = wsample_one(_w, rng);
    for k in (n - 1):-1:1
      @simd for i in Base.OneTo(N)
        @inbounds _w[i] = W[i, k] +
                          logGk(model, X[i, k], X[ai, k + 1], k + 1) +
                          logMk(model, X[ai, k + 1], X[i, k], k + 1);
      end
      normalise_logweights!(_w);
      @inbounds ref[k] = ai = wsample_one(_w, rng);
    end
    nothing;
end

"""
   `get_reference!(x::AbstractVector{P}, st::CPFStorage{P})`

Read the concrete reference trajectory to `x` from `st`.

ASSUMPTIONS:
* `traceback!` has been run before invoking this function and the contents \
of `st` have not changed after that. The results from this function \
WILL BE WRONG, if this has not been done.
* `length(x) == length(st)` (checked, throws error)
"""
@inline function get_reference!(x::AbstractVector{P}, st::CPFStorage{P}) where P
    n = length(st);
    if length(x) != n
        msg = string("length of input reference (", length(x), ") does not ",
                     "match the length of storage (", n, ").");
        throw(ArgumentError(msg));
    end
    X = st.pfst.X; ref = st.ref;
    @simd for k in Base.OneTo(n)
        @inbounds x[k] = X[ref[k], k]; 
    end
    nothing;
end

