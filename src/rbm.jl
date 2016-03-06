
using Base.LinAlg.BLAS
using Distributions
import StatsBase.fit
import StatsBase.coef
import StatsBase: sample, sample!

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

@runonce abstract AbstractRBM{T,V,H}

@runonce type RBM{T,V,H} <: AbstractRBM{T,V,H}
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


# some well-known RBM kinds
BernoulliRBM(n_vis::Int, n_hid::Int; sigma=0.01) =
    RBM(Float64, Degenerate, Bernoulli, n_vis, n_hid; sigma=sigma)

GRBM(n_vis::Int, n_hid::Int; sigma=0.01) =
    RBM(Float64, Normal, Bernoulli, n_vis, n_hid; sigma=sigma)


function Base.show{T,V,H}(io::IO, rbm::RBM{T,V,H})
    n_vis = size(rbm.vbias, 1)
    n_hid = size(rbm.hbias, 1)
    print(io, "RBM{$V,$H}($n_vis, $n_hid)")
end

## utils

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


function sample_hiddens{T,V,H}(rbm::AbstractRBM{T,V,H}, vis::Mat{T})
    means = hid_means(rbm, vis)
    return sample(H, means)
end


function sample_visibles{T,V,H}(rbm::AbstractRBM{T,V,H}, hid::Mat{T})
    means = vis_means(rbm, hid)
    return sample(V, means)
end


function gibbs{T}(rbm::AbstractRBM, vis::Mat{T}; n_times=1)
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

    fe_exp = 1 + exp(rbm.W * vis .+ rbm.hbias)

    # Iterate over fe_exp and shift any 0s or Infs
    # to avoid producing or propagating any Infs
    # into the rest of the calculation.
    for i in eachindex(fe_exp)
        if fe_exp[i] == 0.0
            fe_exp[i] = nextfloat(fe_exp[i])
        end

        if isinf(fe_exp[i])
            if fe_exp[i] > 0.0
                fe_exp[i] = prevfloat(fe_exp[i])
            else
                fe_exp[i] = nextfloat(fe_exp[i])
            end
        end
    end

    fe_log = log(fe_exp)
    Wx_b_log = sum(log(fe_exp), 1)

    return - vb - Wx_b_log
end


function score_samples{T}(rbm::AbstractRBM, vis::Mat{T};
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
    score_row =  n_feat * log(logistic(fe_corrupted - fe))
    return map(Float64, squeeze(score_row', 2))
end

function pseudo_likelihood(rbm::AbstractRBM, X)
    return mean(score_samples(rbm, X))
end


## gradient calculation

function contdiv{T}(rbm::AbstractRBM, vis::Mat{T}, ctx::Dict)
    n_gibbs = @get(ctx, :n_gibbs, 1)
    v_pos, h_pos, v_neg, h_neg = gibbs(rbm, vis, n_times=n_gibbs)
    return v_pos, h_pos, v_neg, h_neg
end


function persistent_contdiv{T}(rbm::AbstractRBM, vis::Mat{T}, ctx::Dict)
    n_gibbs = @get(ctx, :n_gibbs, 1)
    persistent_chain = @get_array(ctx, :persistent_chain, size(vis), vis)
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


function gradient_classic{T}(rbm::RBM, vis::Mat{T}, ctx::Dict)
    sampler = @get_or_create(ctx, :sampler, persistent_contdiv)
    v_pos, h_pos, v_neg, h_neg = sampler(rbm, vis, ctx)
    dW = @get_array(ctx, :dW_buf, size(rbm.W), similar(rbm.W))
    n_obs = size(vis, 2)
    # same as: dW = ((h_pos * v_pos') - (h_neg * v_neg')) / n_obs
    gemm!('N', 'T', T(1 / n_obs), h_neg, v_neg, T(0.0), dW)
    gemm!('N', 'T', T(1 / n_obs), h_pos, v_pos, T(-1.0), dW)
    # gradient for vbias and hbias
    db = squeeze(sum(v_pos, 2) - sum(v_neg, 2), 2) ./ n_obs
    dc = squeeze(sum(h_pos, 2) - sum(h_neg, 2), 2) ./ n_obs
    return dW, db, dc
end


## updating

function grad_apply_learning_rate!{T,V,H}(rbm::RBM{T,V,H}, X::Mat{T},
                                          dtheta::Tuple, ctx::Dict)
    dW, db, dc = dtheta
    lr = @get(ctx, :lr, T(0.1))
    # same as: dW *= lr
    scal!(length(dW), lr, dW, 1)
    scal!(length(db), lr, db, 1)
    scal!(length(dc), lr, dc, 1)
end


function grad_apply_momentum!{T,V,H}(rbm::RBM{T,V,H}, X::Mat{T},
                                     dtheta::Tuple, ctx::Dict)
    dW, db, dc = dtheta
    momentum = @get(ctx, :momentum, 0.9)
    dW_prev = @get_array(ctx, :dW_prev, size(dW), zeros(T, size(dW)))
    # same as: dW += momentum * dW_prev
    axpy!(momentum, dW_prev, dW)
end


function grad_apply_weight_decay!{T,V,H}(rbm::RBM{T,V,H}, X::Mat{T},
                                         dtheta::Tuple, ctx::Dict)
    # The decay penalty should drive all weights toward
    # zero by some small amount on each update.
    dW, db, dc = dtheta
    decay_kind = @get_or_return(ctx, :weight_decay_kind, nothing)
    decay_rate = @get(ctx, :weight_decay_rate,
                      throw(ArgumentError("If using :weight_decay_kind, weight_decay_rate should also be specified")))
    is_l2 = @get(ctx, :l2, false)
    if decay_kind == :l2
        # same as: dW -= decay_rate * W
        axpy!(-decay_rate, rbm.W, dW)
    elseif decay_kind == :l1
        # same as: dW -= decay_rate * sign(W)
        axpy!(-decay_rate, sign(rbm.W), dW)
    end

end

function grad_apply_sparsity!{T,V,H}(rbm::RBM{T,V,H}, X::Mat{T},
                                         dtheta::Tuple, ctx::Dict)
    # The sparsity constraint should only drive the weights
    # down when the mean activation of hidden units is higher
    # than the expected (hence why it isn't squared or the abs())
    dW, db, dc = dtheta
    cost = @get_or_return(ctx, :sparsity_cost, nothing)
    target = @get(ctx, :sparsity_target, throw(ArgumentError("If :sparsity_cost is used, :sparsity_target should also be defined")))
    curr_sparsity = mean(hid_means(rbm, X))
    penalty = cost * (curr_sparsity - target)
    axpy!(-penalty, dW, dW)
    axpy!(-penalty, db, db)
    axpy!(-penalty, dc, dc)
end


function update_weights!(rbm::RBM, dtheta::Tuple, ctx::Dict)
    dW, db, dc = dtheta
    axpy!(1.0, dW, rbm.W)
    rbm.vbias += db
    rbm.hbias += dc
    # save previous dW
    dW_prev = @get_array(ctx, :dW_prev, size(dW), similar(dW))
    copy!(dW_prev, dW)
end


function update_classic!{T}(rbm::RBM, X::Mat{T}, dtheta::Tuple, ctx::Dict)
    # apply gradient updaters. note, that updaters all have
    # the same signature and are thus composable
    grad_apply_learning_rate!(rbm, X, dtheta, ctx)
    grad_apply_momentum!(rbm, X, dtheta, ctx)
    grad_apply_weight_decay!(rbm, X, dtheta, ctx)
    grad_apply_sparsity!(rbm, X, dtheta, ctx)
    # add gradient to the weight matrix
    update_weights!(rbm, dtheta, ctx)
end


## fitting

function fit_batch!{T}(rbm::RBM, X::Mat{T}, ctx = Dict())
    grad = @get_or_create(ctx, :gradient, gradient_classic)
    upd = @get_or_create(ctx, :update, update_classic!)
    dtheta = grad(rbm, X, ctx)
    upd(rbm, X, dtheta, ctx)
    return rbm
end


function fit{T}(rbm::RBM{T}, X::Mat, ctx = Dict{Any,Any}())
    @assert minimum(X) >= 0 && maximum(X) <= 1
    check_options(ctx)
    n_examples = size(X, 2)
    batch_size = @get(ctx, :batch_size, 100)
    batch_idxs = split_evenly(n_examples, batch_size)
    if @get(ctx, :randomize, false)
        batch_idxs = sample(batch_idxs, length(batch_idxs); replace=false)
    end
    n_epochs = @get(ctx, :n_epochs, 10)
    scorer = @get_or_create(ctx, :scorer, pseudo_likelihood)
    reporter = @get_or_create(ctx, :reporter, TextReporter())
    for epoch=1:n_epochs
        epoch_time = @elapsed begin
            for (batch_start, batch_end) in batch_idxs
                # BLAS.gemm! can't handle sparse matrices, so cheaper
                # to make it dense here
                batch = full(X[:, batch_start:batch_end])
                batch = convert(Array{T}, batch)
                fit_batch!(rbm, batch, ctx)
            end
        end
        score = scorer(rbm, X)
        report(reporter, rbm, epoch, epoch_time, score)
    end
    return rbm
end

fit{T}(rbm::RBM, X::Mat{T}; opts...) = fit(rbm, X, Dict(opts))


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


function coef(rbm::RBM; transpose=true)
    return if transpose rbm.W' else rbm.W end
end
# synonyms
weights = coef

hbias(rbm::RBM) = rbm.hbias

vbias(rbm::RBM) = rbm.vbias
