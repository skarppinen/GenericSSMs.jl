# Fallbacks for getting particle type and observation type based on model definition.
"""
   `particle_type(model::GenericSSM)`

Return the particle type associated with the model `model`.
There exists a fallback for this function, which calls `M1` and checks the
type of the output.
"""
function particle_type(model::GenericSSM) 
    rng = MersenneTwister(1); 
    typeof(M1(model, rng));
end 

"""
   `observation_type(model::GenericSSM)`

Return the observation type associated with the model `model`.
There exists a fallback for this function, which calls `gk` and checks the
type of the output.
"""
function observation_type(model::GenericSSM) 
    rng = MersenneTwister(1);
    typeof(gk(model, M1(model, rng), 1, rng));
end
