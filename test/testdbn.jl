using Boltzmann
using MNIST

X, y = traindata()
X = X[:, 1:1000]                     # take only 1000 observations for speed
X = X / (maximum(X) - (minimum(X)))  # normalize to [0..1]

layers = [("input", GRBM(784, 256)),
          ("hid1", BernoulliRBM(256, 100)),
          ("hid2", BernoulliRBM(100, 100))]
dbn = DBN(layers)
fit(dbn, X)
transform(dbn, X)

