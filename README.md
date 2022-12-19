# GenericSSMs.jl

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://skarppinen.github.io/GenericSSMs.jl/dev)

GenericSSMs.jl is a Julia package that provides building blocks for conducting statistical inference of state space models using particle filters. The package is designed to be small and modular, attempting to provide sensible and reasonably fast, non-allocating primitives for writing particle filtering algorithms.

The main features of the package are:
- API for defining generic state space models
- Standard and conditional particle filtering
- Tracebacking strategies for the conditional particle filter
- Prediction from state space models
- Unconditional simulation from state space models
- Resampling API

An introduction and documentation are provided in [the separate documentation pages](https://skarppinen.github.io/GenericSSMs.jl). 

## Disclaimer

GenericSSMs.jl has been created mainly by the author for himself and may change unexpectedly from version to version. 
While care has been taken to implement the package, bugs are possible. Furthermore, the current documentation pages are brief and may appear vague in parts. If you have any suggestions or need help with the package, please do not hesitate to contact me.

## Author

Santeri Karppinen (skarppinen 'at' iki 'dot' fi) 

## License

MIT
