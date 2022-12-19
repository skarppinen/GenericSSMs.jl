using GenericSSMs
using Resamplings
using Random

## Test normalising constant evaluation using simple model
## of SequentialMonteCarlo.jl.
struct TestModel <: GenericSSM end
function GenericSSMs.M1(model::TestModel, rng)::Float64 
   randn(rng);
end
function GenericSSMs.Mk(model::TestModel, x::Float64, k::Integer, rng)::Float64
    1.5 * x + 0.5 * randn(rng);
end
# Note important to add constants here, otherwise won't match with KF.
function GenericSSMs.logG1(model::TestModel, x::Float64)::Float64
    -0.5 * log(pi) - x * x;
end
function GenericSSMs.logGk(model::TestModel, x::Float64, y::Float64, k::Integer)::Float64
    -0.5 * log(pi) - y * y;
end

## Compute normalising constant estimate by particle filter.
model = TestModel();
n = 5;
N = 50_000_000;
rng = Xoshiro(12192022);
resampling = MultinomialResampling(N);
storage = PFStorage(model, N, n);
logZhat_pf = pf_forward_pass!(storage, model, resampling);

## Compute true normalising constant via univariate Kalman filter.
# x_k+1 = T_k x_k + eta_k, eta_k \sim N(0, Q_k)
# y_k = Z_k x_k + eps_k, eps_k \sim N(0, H_k)
function kf(n) 
    μ⁻ = 0.0; μ⁺ = 0.0;
    Σ⁻ = 1.0; Σ⁺ = 0.0;
    T = 1.5;
    Z = 1.0;
    H = 0.5; 
    Q = 0.25;
    logZhat = 0.0;

    for i in Base.OneTo(n)
      v = 0.0 - Z * μ⁻;
      F = Z * Σ⁻ * Z + H
      K = Σ⁻ * Z * inv(F);
      μ⁺ = μ⁻ + K * v;
      Σ⁺ = (1.0 - K * Z) * Σ⁻;
      logZhat += -0.5 * (log(2.0 * pi) + log(F) + v * inv(F) * v);
      
      μ⁻ = T * μ⁺;
      Σ⁻ = T * Σ⁺ * T + Q;
    end
    logZhat;
end
logZhat_kf = kf(n);

println("Estimate of normalising constant (PF): $logZhat_pf");
println("True normalising constant (KF): $logZhat_kf");
