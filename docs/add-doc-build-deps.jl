using Pkg; 
Pkg.activate(); 
Pkg.add("Documenter");
Pkg.add(url = "https://github.com/skarppinen/Resamplings.jl.git"); 
Pkg.add(url = "https://github.com/skarppinen/GenericSSMs.jl.git"); 
Pkg.instantiate()'

