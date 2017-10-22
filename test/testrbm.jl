using Distributions


const DISTRIBUTIONS = [Degenerate, Gaussian, Bernoulli]

# The role of these tests is to ensure that the
# the general functionality always works and
# doesn't result in NaN or Inf weights.
# Not that the RBM performance is still good.
if :integration in TEST_GROUPS
    @testset "RBM Integration" begin
        n_vis = 100
        n_hid = 10

        for T in [Float32, Float64]
            rbm = RBM(T, Gaussian, Bernoulli, n_vis, n_hid)
            Boltzmann.test(rbm; debug=true)
        end
    end
end

if :benchmark in TEST_GROUPS
    @testset "RBM Benchmark" begin
        n_vis = 100
        n_hid = 10

        suite = BenchmarkGroup()

        for T in [Float32, Float64]
            rbm = RBM(T, Gaussian, Bernoulli, n_vis, n_hid)
            suite = benchmark!(rbm, suite; debug=true)
        end

        tune!(suite)
        results = run(suite, verbose=true, seconds=10)
    end
end
