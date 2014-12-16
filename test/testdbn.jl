using Boltzmann
using MNIST

X, y = traindata()

dbn = DBN([784, 100, 100]; vis_type=BernoulliRBM)
fit(dbn, X)

