
module Boltzmann

export BernoulliRBM,
       GRBM,
       DBN,
       fit,
       transform,
       generate,
       components

include("rbm.jl")
include("dbn.jl")

end
