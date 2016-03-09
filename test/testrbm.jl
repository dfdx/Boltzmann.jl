using Distributions


const DISTRIBUTIONS = [Degenerate, Gaussian, Bernoulli]

# The role of these tests is to ensure that the
# the general functionality always works and
# doesn't result in NaN or Inf weights.
# Not that the RBM performance is still good.
if :integration in TEST_GROUPS
    @testset "RBM Integration" begin
        n_vis = 10
        n_hid = 5
        n_obs = 50

        for T in [Float32, Float64]
            DX = rand(T, n_vis, n_obs)
            SX = map(T, sprand(n_vis, n_obs, 0.1))

            for X in Any[DX, SX]
                for V in DISTRIBUTIONS
                    for H in DISTRIBUTIONS
                        println("\n$(typeof(X)) - ($V, $H)")
                        println("-------------------------------------------")
                        rbm = RBM(T, V, H, n_vis, n_hid)

                        # default arguments
                        println("\nDefault")
                        fit(rbm, X)

                        # weight decay
                        @test_throws ArgumentError fit(rbm, X, weight_decay_kind=:l1)
                        println("\nL1 Weight Decay")
                        fit(rbm, X, weight_decay_kind=:l1, weight_decay_rate=0.01)

                        println("\nL2 Weight Decay")
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
                end
            end
        end
    end
end

if :benchmark in TEST_GROUPS
    @testset "RBM Benchmark" begin
        n_vis = 1000
        n_hid = 200
        n_obs = 300

        df = DataFrame()

        for T in [Float32, Float64]
            DX = rand(T, n_vis, n_obs)
            SX = map(T, sprand(n_vis, n_obs, 0.1))
            rbm = RBM(T, Bernoulli, Bernoulli, n_vis, n_hid)

            for X in Any[DX, SX]
                # For now we're just benchmarking the fit method
                # but we could also get transform, generate, etc
                fit_func() = fit(rbm, X)
                result = benchmark(fit_func, "fit", "$(typeof(X))", 10)

                if isempty(df)
                    df = result
                else
                    df = vcat(df, result)
                end
            end
        end

        # For now just print the output, but we may want load a CSV containing
        # existing master benchmarks and compare the two here.
        println(df)
    end
end
