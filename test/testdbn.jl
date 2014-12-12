using HDF5, Boltzmann

f = h5open("/Users/jfsantos/.julia/v0.3/Mocha/examples/mnist/data/train.hdf5")
data = read(f["data"])
X = convert(Matrix{Float64}, reshape(squeeze(data, 3), (28*28, 60000)))

dbn = DBN([784, 100, 100]; vis_type=BernoulliRBM)
fit(dbn, X)

