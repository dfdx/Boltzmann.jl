
include("utils.jl")

using Base.LinAlg.BLAS
using Distributions
import StatsBase.fit

typealias Mat{T} AbstractArray{T, 2}
typealias Vec{T} AbstractArray{T, 1}
typealias Gaussian Normal


"""
Pseudo distribution that returns mean as is during sampling, i.e.

sample(MeanDistr, means) = means
"""
type MeanDistr <: Distribution{Distributions.Univariate,
                               Distributions.Discrete}
end


## types

abstract AbstractRBM

@runonce type RBM{V,H} <: AbstractRBM
    W::Matrix{Float64}         # matrix of weights between vis and hid vars
    vbias::Vector{Float64}     # biases for visible variables
    hbias::Vector{Float64}     # biases for hidden variables
end

function RBM(V::Type, H::Type,
             n_vis::Int, n_hid::Int; sigma=0.01)
    RBM{V,H}(rand(Normal(0, sigma), (n_hid, n_vis)),
             zeros(n_vis), zeros(n_hid))
end

function Base.show{V,H}(io::IO, rbm::RBM{V,H})
    n_vis = size(rbm.vbias, 1)
    n_hid = size(rbm.hbias, 1)
    print(io, "RBM{$V,$H}($n_vis, $n_hid)")
end

## utils

function logistic(x)
    return 1 ./ (1 + exp(-x))
end


function hid_means(rbm::RBM, vis::Mat{Float64})
    p = rbm.W * vis .+ rbm.hbias
    return logistic(p)
end


function vis_means(rbm::RBM, hid::Mat{Float64})
    p = rbm.W' * hid .+ rbm.vbias
    return logistic(p)
end


## samping

function sample(::Type{MeanDistr}, means::Mat{Float64})
    return means
end

function sample(::Type{Bernoulli}, means::Mat{Float64})
    return float(rand(size(means)) .< means)
end


function sample(::Type{Gaussian}, means::Mat{Float64})
    sigma2 = 1                   # using fixed standard diviation
    samples = zeros(size(means))
    for j=1:size(means, 2), i=1:size(means, 1)
        samples[i, j] = rand(Normal(means[i, j], sigma2))
    end
    return samples
end


function sample_hiddens{V,H}(rbm::RBM{V, H}, vis::Mat{Float64})
    means = hid_means(rbm, vis)
    return sample(H, means)
end


function sample_visibles{V,H}(rbm::RBM{V,H}, hid::Mat{Float64})
    means = vis_means(rbm, hid)
    return sample(V, means)
end


function gibbs(rbm::RBM, vis::Mat{Float64}; n_times=1)
    v_pos = vis
    h_pos = sample_hiddens(rbm, v_pos)
    v_neg = sample_visibles(rbm, h_pos)
    h_neg = sample_hiddens(rbm, v_neg)
    for i=1:n_times-1
        v_neg = sample_visibles(rbm, h_neg)
        h_neg = sample_hiddens(rbm, v_neg)
    end
    return v_pos, h_pos, v_neg, h_neg
end


## scoring

function free_energy(rbm::RBM, vis::Mat{Float64})
    vb = sum(vis .* rbm.vbias, 1)
    Wx_b_log = sum(log(1 + exp(rbm.W * vis .+ rbm.hbias)), 1)
    return - vb - Wx_b_log
end


function score_samples(rbm::AbstractRBM, vis::Mat{Float64};
                       sample_size=10000)
    if issparse(vis)
        # sparse matrices may be infeasible for this operation
        # so using only little sample
        cols = rand(1:size(vis, 2), sample_size)
        vis = full(vis[:, cols])
    end
    n_feat, n_samples = size(vis)
    vis_corrupted = copy(vis)
    idxs = rand(1:n_feat, n_samples)
    for (i, j) in zip(idxs, 1:n_samples)
        vis_corrupted[i, j] = 1 - vis_corrupted[i, j]
    end
    fe = free_energy(rbm, vis)
    fe_corrupted = free_energy(rbm, vis_corrupted)
    return n_feat * log(logistic(fe_corrupted - fe))
end


## gradient calculation

function contdiv(rbm::RBM, vis::Mat{Float64}, config::Dict)
    n_gibbs::Int = @get_or_create(config, :n_gibbs, 1)
    v_pos, h_pos, v_neg, h_neg = gibbs(rbm, vis, n_times=n_gibbs)
    return v_pos, h_pos, v_neg, h_neg
end


## function persistent_contdiv(rbm::RBM, vis::Mat{Float64},
##                             n_gibbs::Int)
    ## if size(rbm.persistent_chain) != size(vis)
    ##     # persistent_chain not initialized or batch size changed, re-initialize
    ##     rbm.persistent_chain = vis
    ## end
    ## # take positive samples from real data
    ## v_pos, h_pos, _, _ = gibbs(rbm, vis)
    ## # take negative samples from "fantasy particles"
    ## rbm.persistent_chain, _, v_neg, h_neg = gibbs(rbm, vis, n_times=n_gibbs)
    ## return v_pos, h_pos, v_neg, h_neg
## end


function gradient_classic(rbm::RBM, vis::Mat{Float64}, config::Dict)
    dW = @get_or_create(config, :dW_buf, similar(rbm.W))
    sampler = @get_or_create(config, :sampler, contdiv)
    v_pos, h_pos, v_neg, h_neg = sampler(rbm, vis, config)
    # same as: dW = (h_pos * v_pos') - (h_neg * v_neg')
    gemm!('N', 'T', 1.0, h_neg, v_neg, 0.0, dW)
    gemm!('N', 'T', 1.0, h_pos, v_pos, -1.0, dW)
    # vbias, hbias
    # TODO: does it makes sence to cache them in `config` as well?
    db = sum(v_pos, 2) - sum(v_neg, 2)
    dc = sum(h_pos, 2) - sum(h_neg, 2)
    return dW, db, dc
end


## updating

function update_grad_learning_rate!(dW::Mat{Float64}, config::Dict)
    lr = @get(config, :lr, 0.1)
    # lr = lr / size(v_pos,2) # TODO: how to apply it without breaking
                              # signature?
    # same as: dW *= lr
    scal!(length(dW), lr, dW, 1)
end

function update_grad_momentum!(dW::Mat{Float64}, config::Dict)
    momentum = @get(config, :momentum, 0.9)
    dW_prev = @get_or_create(config, :dW_prev, copy(dW))
    # same as: dW += momentum * dW_prev
    axpy!(rbm.momentum, rbm.dW_prev, dW)
end

function update_weights!(rbm::RBM, dW::Mat{Float64})
    # TODO: db, dc
    # rbm.vbias += vec(lr * (sum(v_pos, 2) - sum(v_neg, 2)))
    # rbm.hbias += vec(lr * (sum(h_pos, 2) - sum(h_neg, 2)))
    axpy!(1.0, dW, rbm.W)
    # save previous dW
    dW_prev = @get_or_create(config, :dW_prev, similar(dW))
    copy!(rbm.dW_prev, dW)
end


function update_classic!(rbm::RBM, dW::Mat{Float64}, config::Dict)
    # apply gradient updaters. note, that updaters all have
    # the same signature and are essentially composable
    update_grad_learning_rate!(dW, config)
    update_grad_momentum!(dW, config)
    # add gradient to the weight matrix
    update_weights!(rbm, dW)
end


## fitting

function fit_batch!(rbm::RBM, vis::Mat{Float64}; config = Dict())
    grad = @get_or_create(config, :gradient, gradient_classic)
    upd = @get_or_create(config, :update, update_classic!)
    dW, db, dc = grad(rbm, vis, config)
    upd(rbm, dW, config)
    return rbm
end


## function fit(rbm::RBM, X::Mat{Float64};
##              persistent=true, lr=0.1, n_iter=10, batch_size=100, n_gibbs=1)
##     @assert minimum(X) >= 0 && maximum(X) <= 1
##     n_samples = size(X, 2)
##     n_batches = @compat Int(ceil(n_samples / batch_size))
##     w_buf = zeros(size(rbm.W))
##     for itr=1:n_iter
##         tic()
##         for i=1:n_batches
##             batch = X[:, ((i-1)*batch_size + 1):min(i*batch_size, end)]
##             batch = full(batch)
##             fit_batch!(rbm, batch, persistent=persistent,
##                        buf=w_buf, n_gibbs=n_gibbs)
##         end
##         toc()
##         pseudo_likelihood = mean(score_samples(rbm, X))
##         info("Iteration #$itr, pseudo-likelihood = $pseudo_likelihood")
##     end
##     return rbm
## end

function fit(rbm::RBM, X::Matrix{Float64};
             gradient=persistent_contdiv,update! = default_update!, opts...)
    println(Dict{Symbol,Any}(opts))
end


## operations on learned RBM

function transform(rbm::RBM, X::Mat{Float64})
    return hid_means(rbm, X)
end


function generate(rbm::RBM, vis::Vec{Float64}; n_gibbs=1)
    return gibbs(rbm, reshape(vis, length(vis), 1); n_times=n_gibbs)[3]
end

function generate(rbm::RBM, X::Mat{Float64}; n_gibbs=1)
    return gibbs(rbm, X; n_times=n_gibbs)[3]
end


function components(rbm::AbstractRBM; transpose=true)
    return if transpose rbm.W' else rbm.W end
end
# synonym
weights(rbm::AbstractRBM; transpose=true) = components(rbm, transpose)





function live()
    X = rand(20, 10)
    rbm = RBM(MeanDistr, Bernoulli, 20, 10)

end
