# Examples

## Noisy AR(1) model 

The following example showcases the main features of GenericSSMs.jl for 
a simple state-space model called the "noisy AR(1) model":

```math
	\begin{aligned}
	x_1 &\sim N(0, \sigma_1^2) \\
	x_k &\sim N(\rho x_{k-1}, \sigma_x^2) \ \text{ for } k \geq 2 \\
	y_k &\sim N(x_k, \sigma_y^2) \ \text{ for } k \geq 1, \\
	\end{aligned}
```
with parameters ``\theta = (\sigma_1, \sigma_x, \sigma_y, \rho)``.

```
using GenericSSMs
using Resamplings # For resampling algorithms.
using Random
using Distributions

### Model struct definition.
# - y: observed data points.
# - θ: values of parameters.
struct NoisyarModel{ParamType} <: GenericSSM
   y::Vector{Float64}
   θ::ParamType
end

# Construct an instance of model struct with observed data
# and desired parameters. Here, could also use a mutable container
# for parameters, if need be.
y = [1.0, 2.0, 3.0];
θ = (σx = 1.0, σy = 1.0, ρ = 0.8, σ1 = 1.0); 
model = NoisyarModel(y, θ); # Instance of model.

### Running particle filter. 
# First define enough of the interface to run PF.
# NOTE: `(; ...) = model` accesses fields of `model`.
function GenericSSMs.M1(model::NoisyarModel, rng)::Float64
   (; θ) = model;
   rand(rng, Normal(0.0, θ.σ1));
end	
function GenericSSMs.logG1(model::NoisyarModel, x::Float64)
   (; θ, y) = model;
   logpdf(Normal(x, θ.σy), y[1]);
end
function GenericSSMs.Mk(model::NoisyarModel, prev::Float64, k::Integer, rng)
   (; θ, y) = model; 
   rand(rng, Normal(prev, θ.σx));
end
function GenericSSMs.logGk(model::NoisyarModel, prev::Float64, cur::Float64, k::Integer)
   (; θ, y) = model;
   logpdf(Normal(cur, θ.σy), y[k]);
end
N = 16; # Number of particles.
n = length(y); # Number of time points.

# Define resampling (this uses Resamplings.jl)
resampling = SystematicResampling(N; order = :none, randomisation = :none);

# Allocate storage object for particle filtering. 
storage = PFStorage(model, N, n);  

# Set RNG. (optional)
rng = Xoshiro(256);

# Run particle filter. 
pf_forward_pass!(storage, model, resampling); # Using GLOBAL_RNG.
pf_forward_pass!(storage, model, resampling, rng); # Using specified `rng`.

### Running conditional particle filter (CPF).
# Define more of the interface in order to be able to do CPF.
function GenericSSMs.logMk(model::NoisyarModel, cur::Float64, prev::Float64, k::Integer)
   (; θ) = model;
   logpdf(Normal(θ.rho * prev, θ.σx), cur);
end
# Create storage object.
storage = CPFStorage(model, N, n);

# Define conditional resampling (uses Resamplings.jl).
resampling = SystematicResampling(N; intent = :conditional);

# Initialise reference trajectory for CPF (with or without rng).
initialise_reference!(storage, model, resampling);
initialise_reference!(storage, model, resampling, rng);

# Do CPF forward pass + ancestor trace.
cpf_forward_pass!(storage, model, resampling, rng)

# Traceback using ancestor tracing.
traceback!(storage, model, AncestorTracing); # .. or just `traceback!(storage, AncestorTracing)`

# Read reference trajectory from storage.
P = particle_type(model);
traj = Vector{P}(undef, length(storage));
get_reference!(traj, storage); # `traj` is written to.

# Do CPF forward pass + backward sampling.
initialise_reference!(storage, model, resampling, rng);
cpf_forward_pass!(storage, model, resampling, rng);
traceback!(storage, model, BackwardSampling); 
get_reference!(traj, storage);

### Unconditional simulation.
# Let's define more of the interface to be able to simulate.
# Note that for the particular Feynman-Kac model used, `m1` and `mk`
# may be defined in terms of `M1` and `Mk` but in general this is not the case.
function GenericSSMs.m1(model::NoisyarModel, rng)
   GenericSSMs.M1(model, rng);
end
function GenericSSMs.mk(model::NoisyarModel, prev::Float64, k::Integer, rng)
   GenericSSMs.Mk(model, prev, k, rng);
end
function GenericSSMs.gk(model::NoisyarModel, cur::Float64, k::Integer, rng)
   (; θ) = model;
   rand(rng, Normal(cur, θ.σy));
end

# Simulate 10 states / observations / states and observations 
dest = simulate(10, model, Level{:state});
dest = simulate(10, model, Level{:obs});
dest = simulate(10, model, Level{(:state, :obs)});

# Simulation using preallocated vectors. 
dest = Vector{Float64}(undef, 10);
simulate!(dest, model, Level{:state});
simulate!(dest, model, Level{:state}, rng);

dest = Vector{Float64}(undef, 10);
simulate!(dest, model, Level{:obs});
simulate!(dest, model, Level{:obs}, rng);

dest = Vector{Tuple{Float64, Float64}}(undef, 10);
simulate!(dest, model, Level{(:state, :obs)});
simulate!(dest, model, Level{(:state, :obs)}, rng);

### Prediction.
storage = PFStorage(model, N, n); # First define storage.

# Run PF to initialise `storage` for prediction.
pf_forward_pass!(storage, model, resampling, rng); 

# Then do prediction (at different levels, with or without rng):
predst = predict(storage, model, Level{(:state, :obs)}; nahead = 10, nsim = 2000)
predst = predict(storage, model, Level{:state}; nahead = 10, nsim = 2000)
predst = predict(storage, model, Level{:obs}; nahead = 10, nsim = 2000)
predst = predict(storage, model; nahead = 10, nsim = 2000);
predst = predict(storage, model, Level{(:state, :obs)}, rng; nahead = 10, nsim = 2000)
predst = predict(storage, model, Level{:state}, rng; nahead = 10, nsim = 2000)
predst = predict(storage, model, Level{:obs}, rng; nahead = 10, nsim = 2000)

# Prediction with preallocated PredictStorage. `n` stands for time index at which to start.
predst = PredictStorage(storage, model, Level{:obs}; nahead = 10, nsim = 2000, n = 3);
predict!(predst, model);
predict!(predst, model, rng);
predst = PredictStorage(storage, model, Level{:state}; nahead = 10, nsim = 2000);
predict!(predst, model);
predict!(predst, model, rng);
predst = PredictStorage(storage, model, Level{(:state, :obs)}; nahead = 10, nsim = 2000, n = 1);
predict!(predst, model);
predict!(predst, model, rng);
```
