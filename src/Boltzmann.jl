
module Boltzmann

export RBM,
       BernoulliRBM,
       GRBM,
       ## ConditionalRBM,
       DBN,
       DAE,
       Bernoulli,
       Gaussian,
       Degenerate,
       fit,
       transform,
       generate,
       predict,
       coef,
       weights,
       hbias,
       vbias,
       unroll,
       save_params,
       load_params

include("core.jl")

end
