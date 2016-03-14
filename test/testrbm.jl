using Distributions


const DISTRIBUTIONS = [Degenerate, Gaussian, Bernoulli]

# The role of these tests is to ensure that the
# the general functionality always works and
# doesn't result in NaN or Inf weights.
# Not that the RBM performance is still good.
if :integration in TEST_GROUPS
    @testset "RBM Integration" begin
        n_vis = 300
        n_hid = 10

        for T in [Float32, Float64]
            rbm = RBM(T, Gaussian, Bernoulli, n_vis, n_hid)
            Boltzmann.test(rbm; debug=true)
        end
    end
end

if :benchmark in TEST_GROUPS
    @testset "RBM Benchmark" begin
        n_vis = 300
        n_hid = 10

        results = DataFrame()
        for T in [Float32, Float64]
            rbm = RBM(T, Gaussian, Bernoulli, n_vis, n_hid)
            df = Boltzmann.benchmark(rbm; debug=true)
            df[:Type] = fill(T, size(df, 1))
            vcat(results, df)
        end

        # For now just print the output, but we may want load a CSV containing
        # existing master benchmarks and compare the two here.
        println(results)
    end
end
