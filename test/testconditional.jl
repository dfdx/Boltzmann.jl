using Boltzmann
using Distributions
using Base.Test

X = rand(1000, 200)


function conditional_smoke_test()
    model = ConditionalRBM(Bernoulli, Bernoulli, 250, 150; steps=3)
    fit(model, X; n_iter=100, n_gibbs=5)
    forecast = predict(model, X[251:1000, 1]; n_gibbs=5)
end

conditional_smoke_test()
