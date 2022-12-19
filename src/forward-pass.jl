""" 
   `use_resampling_backend(m::Module; eval::Bool = true)`
   
Function registers Module `m` as the backend from which to lookup 
implementations for `GenericSSMs.resample!` and `GenericSSMs.conditional_resample!`.
Of course, it is implicitly assumed that said module contains the required
functions with the correct resampling API used in GenericSSMs.jl.
(see `?GenericSSMs.resample!` and `?GenericSSMs.conditional_resample!`).
Resampling related MethodErrors will occur if this is not the case.
Calling this function with `eval = false` returns the code that is used internally
to set the resampling backend.

Currently, a single resampling backend at a time can be used.
"""
function use_resampling_backend(m::Module; evaluate::Bool = true)
   IV = AbstractVector{<: Integer};
   FV = AbstractVector{<: AbstractFloat};
   I = Integer;
   code = quote 
       @inline function GenericSSMs.resample!(res, ind::$IV, 
                                              w::$FV, rng = Random.GLOBAL_RNG)
           $m.resample!(res, ind, w, rng)
       end
       @inline function GenericSSMs.conditional_resample!(res, ind::$IV, w::$FV, 
                                                k::$I, i::$I, rng = Random.GLOBAL_RNG)
           $m.conditional_resample!(res, ind, w, k, i, rng);  
       end
   end
   if evaluate 
       eval(code); return;
   else
       code;
   end
end

"""
   `pf_forward_pass!(st::PFStorage, model::GenericSSM, resampling[, rng = Random.GLOBAL_RNG])`

Run the standard particle filter for model `model` using
storage `st` and resampling `resampling` that implements 
`resample!` (see `?GenericSSMs.resample!`). 
An RNG may be specified as the last argument.

The return value is the logarithm of the normalising constant
estimate of the particle filter.
"""
function pf_forward_pass!(
				 st::PFStorage, 
				 model::GenericSSM,
				 resampling,
             rng = Random.GLOBAL_RNG)

    X = st.X; W = st.W; A = st.A; _w = st._w; 
    N, n = size(st); 
	 logZhat = 0.0;
    
    # Simulate from initial distribution.
    for i in Base.OneTo(N)
       @inbounds X[i, 1] = M1(model, rng);
    end
    # Compute initial weights.
    for i in Base.OneTo(N) 
        @inbounds _w[i] = W[i, 1] = logG1(model, X[i, 1]);
    end
    logZhat += normalise_logweights!(_w);
    for k in 2:n
       # Resample.
       a = view(A, :, k - 1);
       resample!(resampling, a, _w, rng);
		 # Propagate.
		 for i in Base.OneTo(N) 
		 	@inbounds X[i, k] = Mk(model, X[a[i], k - 1], k, rng);
       end
		 # Compute weights.
		 for i in Base.OneTo(N) 
         @inbounds _w[i] = W[i, k] = logGk(model, X[a[i], k - 1], X[i, k], k);
       end
       logZhat += normalise_logweights!(_w);
    end
    logZhat; 
end

"""
   `pf_forward_pass!(st::CPFStorage, model::GenericSSM, resampling[, rng = Random.GLOBAL_RNG])`

Additional method for `pf_forward_pass!` that allows running the standard particle filter
with the `CPFStorage` storage object. 
This method does not alter the field `st.ref` since it is not needed in standard particle filtering. 
"""
@inline function pf_forward_pass!(
            st::CPFStorage,
            model::GenericSSM,
            resampling,
            rng = Random.GLOBAL_RNG)
   pf_forward_pass!(st.pfst, model, resampling, rng);
end

"""
   `cpf_forward_pass!(st::CPFStorage, model::GenericSSM, resampling[, rng = Random.GLOBAL_RNG])`

Run the forward pass of the conditional particle filter using \
storage `st` for model `model` with resampling `resampling` \
that implements `conditional_resample!` (see `?GenericSSMs.conditional_resample!`). 
An RNG may be specified as the last argument.

ASSUMPTIONS:
* the reference trajectory has been initialised to `st` \
(fields `st.ref` and `st.pfst` have been populated). \
Results output by this function WILL BE WRONG, if this has not been done. \
Use `initialise_reference!` to properly initialise reference. 
"""
function cpf_forward_pass!(
      st::CPFStorage, 
      model::GenericSSM,
      resampling,
      rng = Random.GLOBAL_RNG)

   X = st.pfst.X; W = st.pfst.W; A = st.pfst.A; 
   _w = st.pfst._w; ref = st.ref; 
   N, n = size(st); 

   # Simulate from initial distribution.
   for i in Base.OneTo(N) 
      (i == @inbounds ref[1]) && continue;
      @inbounds X[i, 1] = M1(model, rng); 
   end
   # Compute initial weights.
   for i in Base.OneTo(N)
      @inbounds _w[i] = W[i, 1] = logG1(model, X[i, 1]);
   end
   normalise_logweights!(_w);

   for k in 2:n
      # Resample and propagate surviving particles.
      a = view(A, :, k - 1);
      @inbounds conditional_resample!(resampling, a, _w, ref[k], ref[k - 1]);
      for i in Base.OneTo(N)
           (i == @inbounds ref[k]) && continue;
           @inbounds X[i, k] = Mk(model, X[a[i], k - 1], k, rng);
        end
        # Compute weights.
        for i in Base.OneTo(N) 
            @inbounds _w[i] = W[i, k] = logGk(model, X[a[i], k - 1], X[i, k], k);
        end
        normalise_logweights!(_w);
    end
    nothing;
end

"""
   `initialise_reference!(st::CPFStorage, model::GenericSSM, resampling[, rng = Random.GLOBAL_RNG])`

Initialise the reference trajectory for running the conditional particle
filter forward pass (`cpf_forward_pass!`).
"""
function initialise_reference!(
      st::CPFStorage,
      model::GenericSSM,
      resampling,
      rng = Random.GLOBAL_RNG)
   pf_forward_pass!(st, model, resampling, rng);
   traceback!(st, AncestorTracing, rng);
end
