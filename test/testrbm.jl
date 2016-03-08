
using Boltzmann
using Distributions
using Base.Test

n_vis = 200
n_hid = 50
n_obs = 1000


const DISTRIBUTIONS = [Degenerate, Gaussian, Bernoulli]

function run_tests(T, V, H, X)
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

    @test !any(isnan, rbm.W)
    @test !any(isnan, rbm.vbias)
    @test !any(isnan, rbm.hbias)
    @test !any(isinf, rbm.W)
    @test !any(isinf, rbm.vbias)
    @test !any(isinf, rbm.hbias)

    # sparsity
    # TODO
end

for T in [Float32, Float64]
    DX = rand(T, n_vis, n_obs)
    SX = map(T, sprand(n_vis, n_obs, 0.1))
    for X in Any[DX, SX]
        for V in DISTRIBUTIONS
            for H in DISTRIBUTIONS
                run_tests(T, V, H, X)
            end
        end
    end
end
