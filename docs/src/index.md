# Introduction 

GenericSSMs.jl is a Julia package that provides building blocks for conducting statistical inference of [state-space models (SSMs)](ssms.md) using particle filters. 
The package is designed to be small and modular, attempting to provide sensible and reasonably fast, non-allocating primitives for writing particle filtering algorithms. 

GenericSSMs.jl assumes that the user is somewhat familiar with state-space models and their Feynman-Kac representations. This documentation provides [a very short introduction](ssms.md) to these topics. The references therein can be used to find more information. 

The main features of GenericSSMs.jl are:

- an [interface](interface.md) for defining an SSM using a Feynman-Kac representation that may depend on arbitrary data and parameters (a "generic SSM")
- [particle filtering](use-cases.md#Particle-filtering) & [conditional particle filtering](use-cases.md#Conditional-particle-filtering) [Andrieu, Doucet, Holenstein (2010)] for generic SSMs
- [tracebacking strategies](use-cases.md#Tracebacking strategies) for the conditional particle filter: 
   - ancestor tracing [Andrieu, Doucet, Holenstein (2010)]
   - backward sampling [Whiteley (2010)]
- [unconditional simulation](use-cases.md#Unconditional-simulation) from generic SSMs at state and/or observation level
- [prediction](use-cases.md#Prediction-from-SSMs) from generic SSMs at state and/or observation level
- a generic [resampling API](resampling-api.md) for user-defined resampling algorithms
   - default resampling algorithms are provided via [Resamplings.jl](https://github.com/skarppinen/Resamplings.jl)


##### References

- _Andrieu, C., Doucet, A., and Holenstein, R. (2010) Particle Markov chain Monte Carlo methods. Journal of the Royal Statistical Society: Series B. 72(3):269-342._
- _Whiteley, N. (2010) Discussion on "Particle Markov chain Monte Carlo methods". Journal of the Royal Statistical Society: Series B. 72(3):306-307._
 
