
module Boltzmann

export RBM,
       ## BernoulliRBM,
       ## GRBM,
       ## ConditionalRBM,
       ## DBN,
       ## DAE,
       Bernoulli,
       Gaussian,
       MeanDistr,
       fit,
       transform,
       generate,
       predict,
       # components,
       weights
       ## unroll,
       ## save_params,
       ## load_params

include("core.jl")

end
