
using Boltzmann
using Distributions
using Base.Test

n_vis = 1000
n_hid = 200

const DISTRIBUTIONS = [Degenerate, Gaussian, Bernoulli]

for T in [Float32, Float64]    
    X = rand(T, n_vis, n_hid)    
    for V in DISTRIBUTIONS
        for H in DISTRIBUTIONS
            println("------------------------------------------------------")
            println("Type=$T, Visible=$V, Hidden=$H")
            println("------------------------------------------------------")
            
            rbm = RBM(T, V, H, n_vis, n_hid)

            # default arguments
            fit(rbm, X)
    

            # weight decay
            @test_throws ArgumentError fit(rbm, X, weight_decay_kind=:l1)
            fit(rbm, X, weight_decay_kind=:l1, weight_decay_rate=0.01)
            fit(rbm, X, weight_decay_kind=:l2, weight_decay_rate=0.01)

            # sparsity
            # TODO
        end
    end    
end
