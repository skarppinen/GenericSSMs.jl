# Installation

To install and use GenericSSMs.jl, run the following commands in the Julia REPL:
```
# Install GenericSSMs and its dependency Resamplings (need to do only once).
import Pkg;
Pkg.add(url = "https://github.com/skarppinen/Resamplings.jl"); 
Pkg.add(url = "https://github.com/skarppinen/GenericSSMs.jl"); 

# Load GenericSSMs. 
using GenericSSMs
```

# Quick start

To define a simple bootstrap filter with GenericSSMs.jl, see the first example in [Examples](examples.md).
