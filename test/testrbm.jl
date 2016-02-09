
using Boltzmann
using Distributions
using Base.Test

## X = rand(1000, 200)


## function brbm_smoke_test()
##     model = BernoulliRBM(1000, 500)
##     fit(model, X)
## end

## function grbm_smoke_test()
##     model = GRBM(1000, 500)
##     fit(model, X)
## end

## function conf_smoke_test()
##     model = RBM(Normal, Normal, 1000, 500)
##     fit(model, X)
## end

## brbm_smoke_test()
## grbm_smoke_test()
## conf_smoke_test()

n_vis = 1000
n_hid = 200
X = rand(n_vis, n_hid)
rbm = RBM(Degenerate, Bernoulli, n_vis, n_hid)

# weight decay
@test_throws ArgumentError fit(rbm, X, weight_decay_kind=:l1)
fit(rbm, X, weight_decay_kind=:l1, weight_decay_rate=0.01)
fit(rbm, X, weight_decay_kind=:l2, weight_decay_rate=0.01)

# sparsity
# TODO

