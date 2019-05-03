using Boltzmann
using Distributions
using Test
using MLDatasets.MNIST


function _get_dataset(X)
    X = round.(X ./ (maximum(X) - minimum(X)))
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
        model = ConditionalRBM(Bernoulli, Bernoulli, 250, 150; steps=3)
        Boltzmann.test(model; input_size=1000, debug=true)
    end
end

if :acceptance in TEST_GROUPS
    @testset "Conditional RBM Acceptance" begin
        X = MNIST.convert2features(traindata()[1])[:, 1:10000]
        train_X = _get_dataset(X)
	    input_size = round(Int, size(train_X, 1) / 2)

        model = ConditionalRBM(Bernoulli, Bernoulli, input_size, 500; steps=1)
        fit(model, train_X; n_epochs=10, n_gibbs=5)

        test_X = _get_dataset(X)
        corrupt_start = input_size + 1
        forecast = predict(model, test_X[corrupt_start:end,:]; n_gibbs=20)

        mae = mean(test_X[1:corrupt_start-1,:] .!= forecast)

        # At a bare minimum the prediction should have an mae
        # less than the amount of noise added to the condition
        # values. In reality, this should be around ~0.05
        # and the likelihood should be move from >-60 to <-31
        println("MAE = $mae")
        @test mae < 0.1
    end
end

if :benchmark in TEST_GROUPS
    @testset "Conditional RBM Benchmark" begin
        n_vis = 250
        n_cond = 750
        n_hid = 200

        suite = BenchmarkGroup()

        for T in [Float32, Float64]
            model = ConditionalRBM(T, Bernoulli, Bernoulli, n_vis, n_hid, n_cond)
            suite = benchmark!(model, suite; input_size=1000, debug=true)
        end

        tune!(suite)
        results = run(suite, verbose=true, seconds=10)
    end
end
