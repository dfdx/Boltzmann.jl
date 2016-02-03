
include("utils.jl")

using Base.LinAlg.BLAS
using Distributions
import StatsBase.fit

typealias Mat{T} AbstractArray{T, 2}
typealias Vec{T} AbstractArray{T, 1}
typealias Gaussian Normal


"""
Distribution with a single possible value. Used e.g. during sampling
to provide stable result equal to provided means:

sample(Degenerate, means) = means
"""
type Degenerate <: Distribution{Distributions.Univariate,
                               Distributions.Discrete}
end


## types

abstract AbstractRBM

@runonce type RBM{T,V,H} <: AbstractRBM
    W::Matrix{T}         # matrix of weights between vis and hid vars
    vbias::Vector{T}     # biases for visible variables
    hbias::Vector{T}     # biases for hidden variables
end

function RBM(T::Type, V::Type, H::Type,
             n_vis::Int, n_hid::Int; sigma=0.01)
    RBM{T,V,H}(map(T, rand(Normal(0, sigma), (n_hid, n_vis))),
             zeros(n_vis), zeros(n_hid))
end

RBM(V::Type, H::Type, n_vis::Int, n_hid::Int; sigma=0.01) =
    RBM(Float64, V, H, n_vis, n_hid; sigma=sigma)


function Base.show{T,V,H}(io::IO, rbm::RBM{T,V,H})
    n_vis = size(rbm.vbias, 1)
    n_hid = size(rbm.hbias, 1)
    print(io, "RBM{$V,$H}($n_vis, $n_hid)")
end

## utils

function logistic(x)
    return 1 ./ (1 + exp(-x))
end


function hid_means{T}(rbm::RBM, vis::Mat{T})
    p = rbm.W * vis .+ rbm.hbias
    return logistic(p)
end


function vis_means{T}(rbm::RBM, hid::Mat{T})
    p = rbm.W' * hid .+ rbm.vbias
    return logistic(p)
end


## samping

function sample{T}(::Type{Degenerate}, means::Mat{T})
    return means
end

function sample{T}(::Type{Bernoulli}, means::Mat{T})
    return map(T, float((rand(size(means)) .< means)))
end


function sample{T}(::Type{Gaussian}, means::Mat{T})
    sigma2 = 1                   # using fixed standard diviation
    samples = zeros(T,size(means))
    for j=1:size(means, 2), i=1:size(means, 1)
        samples[i, j] = T(rand(Normal(means[i, j], sigma2)))
    end
    return samples
end


function sample_hiddens{T,V,H}(rbm::RBM{T,V,H}, vis::Mat{T})
    means = hid_means(rbm, vis)
    return sample(H, means)
end


function sample_visibles{T,V,H}(rbm::RBM{T,V,H}, hid::Mat{T})
    means = vis_means(rbm, hid)
    return sample(V, means)
end


function gibbs{T}(rbm::RBM, vis::Mat{T}; n_times=1)
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

function free_energy{T}(rbm::RBM, vis::Mat{T})
    vb = sum(vis .* rbm.vbias, 1)
    Wx_b_log = sum(log(1 + exp(rbm.W * vis .+ rbm.hbias)), 1)
    return - vb - Wx_b_log
end


function score_samples{T}(rbm::RBM, vis::Mat{T};
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

function pseudo_likelihood(rbm::RBM, X)
    return mean(score_samples(rbm, X))
end


## gradient calculation

function contdiv{T}(rbm::RBM, vis::Mat{T}, config::Dict)
    n_gibbs = @get_or_create(config, :n_gibbs, 1)
    v_pos, h_pos, v_neg, h_neg = gibbs(rbm, vis, n_times=n_gibbs)
    return v_pos, h_pos, v_neg, h_neg
end


function persistent_contdiv{T}(rbm::RBM, vis::Mat{T}, config::Dict)
    n_gibbs = @get_or_create(config, :n_gibbs, 1)
    persistent_chain = @get_or_create(config, :persistent_chain, vis)
    if size(persistent_chain) != size(vis)
        # persistent_chain not initialized or batch size changed
        # re-initialize
        persistent_chain = vis
    end
    # take positive samples from real data
    v_pos, h_pos, _, _ = gibbs(rbm, vis)
    # take negative samples from "fantasy particles"
    persistent_chain, _, v_neg, h_neg = gibbs(rbm, vis, n_times=n_gibbs)
    return v_pos, h_pos, v_neg, h_neg
end


function gradient_classic{T}(rbm::RBM, vis::Mat{T}, config::Dict)
    sampler = @get_or_create(config, :sampler, persistent_contdiv)
    v_pos, h_pos, v_neg, h_neg = sampler(rbm, vis, config)
    dW = @get_or_create(config, :dW_buf, similar(rbm.W))
    # same as: dW = (h_pos * v_pos') - (h_neg * v_neg')
    gemm!('N', 'T', T(1 / size(vis, 2)), h_neg, v_neg, T(0.0), dW)
    gemm!('N', 'T', T(1 / size(vis, 2)), h_pos, v_pos, T(-1.0), dW)
    # gradient for vbias and hbias
    db = squeeze(sum(v_pos, 2) - sum(v_neg, 2), 2) ./ size(vis, 2)
    dc = squeeze(sum(h_pos, 2) - sum(h_neg, 2), 2) ./ size(vis, 2)
    return dW, db, dc
end


## updating

function update_delta_learning_rate!{T}(dtheta::Tuple{Mat{T},Vec{T},Vec{T}},
                                       config::Dict)
    dW, db, dc = dtheta
    lr = @get(config, :lr, T(0.1))
    # same as: dW *= lr
    scal!(length(dW), lr, dW, 1)
end

function update_delta_momentum!{T}(dtheta::Tuple{Mat{T}, Vec{T}, Vec{T}},
                                  config::Dict)
    dW, db, dc = dtheta
    momentum = @get(config, :momentum, 0.9)
    dW_prev = @get_or_create(config, :dW_prev, copy(dW))
    # same as: dW += momentum * dW_prev
    axpy!(momentum, dW_prev, dW)
end

function update_delta_weight_decay!{T}(dtheta::Tuple{Mat{T}, Vec{T}, Vec{T}},
                                       config::Dict)
    # TODO: this is L2 regularization; should we also consider L1?
    dW, db, dc = dtheta
    n_obs = size(dW, 1)
    decay_rate = @get_or_return(config, :weight_decay_rate, nothing) / n_obs
    # same as: dW -= decay_rate * dW
    axpy!(-decay_rate, dW, dW)
end

function update_weights!{T}(rbm::RBM, dtheta::Tuple{Mat{T}, Vec{T}, Vec{T}},
                            config::Dict)
    dW, db, dc = dtheta
    axpy!(1.0, dW, rbm.W)
    rbm.vbias += db
    rbm.hbias += dc
    # save previous dW
    dW_prev = @get_or_create(config, :dW_prev, similar(dW))
    copy!(dW_prev, dW)
end


function update_classic!{T}(rbm::RBM, dtheta::Tuple{Mat{T}, Vec{T}, Vec{T}},
                            config::Dict)
    # apply gradient updaters. note, that updaters all have
    # the same signature and are essentially composable
    update_delta_learning_rate!(dtheta, config)
    update_delta_momentum!(dtheta, config)
    update_delta_weight_decay!(dtheta, config)
    # add gradient to the weight matrix
    update_weights!(rbm, dtheta, config)
end


## fitting

function fit_batch!{T}(rbm::RBM, vis::Mat{T}, config = Dict())
    grad = @get_or_create(config, :gradient, gradient_classic)
    upd = @get_or_create(config, :update, update_classic!)
    dtheta = grad(rbm, vis, config)
    upd(rbm, dtheta, config)
    return rbm
end


function fit{T}(rbm::RBM, X::Matrix{T}; config = Dict{Any,Any}())
    @assert minimum(X) >= 0 && maximum(X) <= 1
    n_examples = size(X, 2)
    batch_size = @get(config, :batch_size, 100)
    n_batches = Int(ceil(n_examples / batch_size))
    n_epochs = @get(config, :n_epochs, 10)
    scorer = @get_or_create(config, :scorer, pseudo_likelihood)
    reporter = @get_or_create(config, :reporter, msg -> println(msg))
    for epoch=1:n_epochs
        epoch_time = @elapsed begin
            for i=1:n_batches
                batch = X[:, ((i-1)*batch_size + 1):min(i*batch_size, end)]
                batch = full(batch)
                fit_batch!(rbm, batch, config)
            end
        end
        score = scorer(rbm, X)
        # TODO: reporter should get score and elapsed time as parameters
        # and report in whatever way it needs
        reporter("Epoch $epoch: score=$score; " *
                 "time taken=$epoch_time seconds")
    end
    return rbm
end


## operations on learned RBM

function transform{T}(rbm::RBM, X::Mat{T})
    return hid_means(rbm, X)
end


function generate{T}(rbm::RBM, vis::Vec{T}; n_gibbs=1)
    return gibbs(rbm, reshape(vis, length(vis), 1); n_times=n_gibbs)[3]
end

function generate{T}(rbm::RBM, X::Mat{T}; n_gibbs=1)
    return gibbs(rbm, X; n_times=n_gibbs)[3]
end


function components(rbm::RBM; transpose=true)
    return if transpose rbm.W' else rbm.W end
end
# synonym
weights(rbm::AbstractRBM; transpose=true) = components(rbm, transpose)


function live()
    X = rand(Float32, 20, 10)
    rbm = RBM(Float32, Degenerate, Bernoulli, 20, 10)
    fit(rbm, X; config=Dict{Any,Any}(:sampler => contdiv))
end
