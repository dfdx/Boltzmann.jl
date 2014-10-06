
using RBM
using Base.Test

X = rand(1000, 2000)


function brbm_smoke_test()
    model = BernoulliRBM(1000, 500)
    fit!(model, X)
end

function grbm_smoke_test()
    model = GRBM(1000, 500)
    fit!(model, X)
end


brbm_smoke_test()
grbm_smoke_test()
