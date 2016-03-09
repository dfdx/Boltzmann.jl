using Boltzmann
using MNIST

if :integration in TEST_GROUPS
    @testset "Nets Integration" begin
        X, y = traindata()
        X = X[:, 1:1000]                     # take only 1000 observations for speed
        X = X / (maximum(X) - (minimum(X)))  # normalize to [0..1]

        layers = [("vis", GRBM(784, 256)),
                  ("hid1", BernoulliRBM(256, 100)),
                  ("hid2", BernoulliRBM(100, 100))]
        dbn = DBN(layers)
        fit(dbn, X)
        @assert size(transform(dbn, X)) == (100, 1000)

        dae = unroll(dbn)
        transform(dae, X)

        save_params("test.hdf5", dbn)
        save_params("test2.hdf5", dae)
        rm("test.hdf5")
        rm("test2.hdf5")
    end
end

