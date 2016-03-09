using Boltzmann
using Distributions
using Base.Test
using MNIST


function _get_dataset(X)
    X = round(X ./ (maximum(X) - minimum(X)))
    X_c = copy(X)   # the corrupted images

    # Flip 10% of bits in X_c
    flip_count = round(Int, size(X_c, 1) * 0.1)
    for i in 1:size(X_c, 2)
        flips = map(i -> rand(1:size(X_c, 1)), 1:flip_count)
        for j in flips
            item = Bool(X_c[j,i])
            X_c[j,i] = item ? 0.0 : 1.0
        end
    end

    return vcat(X, X_c)
end


if :integration in TEST_GROUPS
    @testset "Conditional RBM Integration" begin
        X = rand(1000, 200)
        model = ConditionalRBM(Bernoulli, Bernoulli, 250, 150; steps=3)
        fit(model, X; n_epochs=10, n_gibbs=5)
        forecast = predict(model, X[251:1000, 1]; n_gibbs=5)
    end
end

if :acceptance in TEST_GROUPS
    @testset "Conditional RBM Acceptance" begin
        train_X = _get_dataset(traindata()[1][:, 1:10000])
        input_size = round(Int, size(train_X, 1) / 2)

        model = ConditionalRBM(Bernoulli, Bernoulli, input_size, 500; steps=1)
        fit(model, train_X; n_epochs=10, n_gibbs=5)

        test_X = _get_dataset(testdata()[1][:, 1:1000])
        corrupt_start = input_size + 1
        forecast = predict(model, test_X[corrupt_start:end,:]; n_gibbs=20)

        mae = mean(test_X[1:corrupt_start-1,:] .!= forecast)

        # At a bare minimum the prediction should have an mae
        # less than the amount of noise added to the condition
        # values. In reality, this should be around ~0.05
        # and the likelihood should be move from >-60 to <-31
        @test mae < 0.1

        @test !any(isnan, model.W)
        @test !any(isnan, model.A)
        @test !any(isnan, model.B)
        @test !any(isinf, model.W)
        @test !any(isinf, model.A)
        @test !any(isinf, model.B)
    end
end

if :benchmark in TEST_GROUPS
    @testset "Conditional RBM Benchmark" begin
        n_vis = 250
        n_cond = 750
        n_hid = 200
        n_obs = 300

        df = DataFrame()

        for T in [Float32, Float64]
            X = rand(T, n_vis + n_cond, n_obs)
            rbm = ConditionalRBM(T, Bernoulli, Bernoulli, n_vis, n_hid, n_cond)

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

        # For now just print the output, but we may want load a CSV containing
        # existing master benchmarks and compare the two here.
        println(df)
    end
end

