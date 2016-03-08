using Boltzmann
using Distributions
using Base.Test
using MNIST


function conditional_smoke_test()
    X = rand(1000, 200)
    model = ConditionalRBM(Bernoulli, Bernoulli, 250, 150; steps=3)
    fit(model, X; n_epochs=10, n_gibbs=5)
    forecast = predict(model, X[251:1000, 1]; n_gibbs=5)
end

function run_mnist()
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

conditional_smoke_test()
run_mnist()
