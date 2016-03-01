
## Julia implementation of a ConditionalRBM from Graham Taylor's PhD Thesis

## Links:
##     Thesis - http://www.cs.nyu.edu/~gwtaylor/thesis/Taylor_Graham_W_200911_PhD_thesis.pdf
##     FCRBM - http://www.cs.toronto.edu/~fritz/absps/fcrbm_icml.pdf


# Input data layout for 2 batches and 3 steps of history
#
# NOTE: in the code below the history is referred to as `cond`
# since technically you can condition on any features not just
# previous time steps.
#
#      batch 1               batch 2
# |--------------------|--------------------|-----
# |  current visible   |  current visible   | ...
# |--------------------|--------------------|-----
# |  history step 1    |  history step 1    | ...
# |  history step 2    |  history step 2    | ...
# |  history step 3    |  history step 3    | ...
# |--------------------|--------------------|-----


import StatsBase: predict


@runonce type ConditionalRBM{T,V,H} <: AbstractRBM{T,V,H}
    W::Matrix{T}  # standard weights
    A::Matrix{T}  # autoregressive params (vis to vis)
    B::Matrix{T}  # hidden params(vis to hid)
    vbias::Vector{T}
    hbias::Vector{T}
    dyn_vbias::Array{T}
    dyn_hbias::Array{T}
end


function ConditionalRBM(T::Type, V::Type, H::Type,
                        n_vis::Int, n_hid::Int, n_cond::Int; sigma=0.01)
    ConditionalRBM{T,V,H}(
        map(T, rand(Normal(0, sigma), (n_hid, n_vis))),
        map(T, rand(Normal(0, sigma), (n_vis, n_cond))),
        map(T, rand(Normal(0, sigma), (n_hid, n_cond))),
        zeros(T, n_vis),
        zeros(T, n_hid),
        zeros(T, n_vis),
        zeros(T, n_hid),
    )
end

function ConditionalRBM(V::Type, H::Type,
                        n_vis::Int, n_hid::Int, n_cond::Int; sigma=0.01)
    ConditionalRBM(Float64, V, H, n_vis, n_hid, n_cond;
                   sigma=sigma)
end

function ConditionalRBM(T::Type, V::Type, H::Type, n_vis::Int, n_hid::Int;
                        steps=5, sigma=0.01)
    ConditionalRBM(T, V, H, n_vis, n_hid, (n_vis * steps);
                   sigma=sigma)
end

function ConditionalRBM(V::Type, H::Type, n_vis::Int, n_hid::Int;
                        steps=5, sigma=0.01)
    ConditionalRBM(Float64, V, H, n_vis, n_hid, (n_vis * steps);
                   sigma=sigma)
end


function Base.show{T,V,H}(io::IO, crbm::ConditionalRBM{T,V,H})
    n_vis = size(crbm.vbias, 1)
    n_hid = size(crbm.hbias, 1)
    n_cond = size(crbm.A, 2)
    print(io, "ConditionalRBM{$T,$V,$H}($n_vis, $n_hid, $n_cond)")
end


function split_vis{T}(crbm::ConditionalRBM, vis::Mat{T})
    vis_size = length(crbm.vbias)

    curr = sub(vis, 1:vis_size, :)
    cond = sub(vis, (vis_size + 1):(vis_size + size(crbm.A, 2)), :)

    return curr, cond
end


function dynamic_biases!{T}(crbm::ConditionalRBM, cond::Mat{T})
    crbm.dyn_vbias = crbm.A * cond .+ crbm.vbias
    crbm.dyn_hbias = crbm.B * cond .+ crbm.hbias
end


function hid_means{T}(crbm::ConditionalRBM, vis::Mat{T})
    p = crbm.W * vis .+ crbm.dyn_hbias
    return logistic(p)
end


function vis_means{T}(crbm::ConditionalRBM, hid::Mat{T})
    p = crbm.W' * hid .+ crbm.dyn_vbias
    return logistic(p)
end


function gradient_classic{T}(crbm::ConditionalRBM, X::Mat{T},
                          ctx::Dict)
    vis, cond = split_vis(crbm, X)
    sampler = @get_or_create(ctx, :sampler, persistent_contdiv)
    v_pos, h_pos, v_neg, h_neg = sampler(crbm, vis, ctx)
    n_obs = size(vis, 2)
    # updating weight matrix W
    dW = @get_array(ctx, :dW_buf, size(crbm.W), similar(crbm.W))
    # same as: dW = ((h_pos * v_pos') - (h_neg * v_neg')) / n_obs
    gemm!('N', 'T', T(1 / n_obs), h_neg, v_neg, T(0.0), dW)
    gemm!('N', 'T', T(1 / n_obs), h_pos, v_pos, T(-1.0), dW)

    # updating vis to vis matrix A
    dA = @get_array(ctx, :dA_buf, size(crbm.A), similar(crbm.A))
    # same as: dW = (h_pos * v_pos') - (h_neg * v_neg')
    gemm!('N', 'T', T(1 / n_obs), v_neg, cond, 0.0, dA)
    gemm!('N', 'T', T(1 / n_obs), v_pos, cond, -1.0, dA)


    # updating hid to hid matrix A
    dB = @get_array(ctx, :dB_buf, size(crbm.B), similar(crbm.B))
    gemm!('N', 'T', T(1 / n_obs), h_neg, cond, 0.0, dB)
    gemm!('N', 'T', T(1 / n_obs), h_pos, cond, -1.0, dB)

    # gradient for vbias and hbias
    db = squeeze(sum(v_pos, 2) - sum(v_neg, 2), 2) ./ n_obs
    dc = squeeze(sum(h_pos, 2) - sum(h_neg, 2), 2) ./ n_obs
    return dW, dA, dB, db, dc
end


function grad_apply_learning_rate!{T,V,H}(crbm::ConditionalRBM{T,V,H},
                                          X::Mat{T},
                                          dtheta::Tuple, ctx::Dict)
    dW, dA, dB, db, dc = dtheta
    lr = @get(ctx, :lr, T(0.1))
    # same as: dW *= lr
    scal!(length(dW), lr, dW, 1)
    scal!(length(dA), lr, dA, 1)
    scal!(length(dB), lr, dB, 1)
    scal!(length(db), lr, db, 1)
    scal!(length(dc), lr, dc, 1)
end


function grad_apply_momentum!{T,V,H}(crbm::ConditionalRBM{T,V,H}, X::Mat{T},
                                     dtheta::Tuple, ctx::Dict)
    dW, dA, dB, db, dc = dtheta
    momentum = @get(ctx, :momentum, 0.9)
    dW_prev = @get_array(ctx, :dW_prev, size(dW), zeros(T, size(dW)))
    # same as: dW += momentum * dW_prev
    axpy!(momentum, dW_prev, dW)
end


function grad_apply_weight_decay!{T,V,H}(rbm::ConditionalRBM{T,V,H},
                                         X::Mat{T},
                                         dtheta::Tuple, ctx::Dict)
    # The decay penalty should drive all weights toward
    # zero by some small amount on each update.
    dW, dA, dB, db, dc = dtheta
    decay_kind = @get_or_return(ctx, :weight_decay_kind, nothing)
    decay_rate = @get(ctx, :weight_decay_rate,
                      throw(ArgumentError("If using :weight_decay_kind, weight_decay_rate should also be specified")))
    is_l2 = @get(ctx, :l2, false)
    if decay_kind == :l2
        # same as: dW -= decay_rate * W
        axpy!(-decay_rate, rbm.W, dW)
        axpy!(-decay_rate, rbm.A, dA)
        axpy!(-decay_rate, rbm.B, dB)
    elseif decay_kind == :l1
        # same as: dW -= decay_rate * sign(W)
        axpy!(-decay_rate, sign(rbm.W), dW)
        axpy!(-decay_rate, sign(rbm.A), dA)
        axpy!(-decay_rate, sign(rbm.B), dB)
    end
end


function grad_apply_sparsity!{T,V,H}(rbm::ConditionalRBM{T,V,H}, X::Mat{T},
                                     dtheta::Tuple, ctx::Dict)
    # The sparsity constraint should only drive the weights
    # down when the mean activation of hidden units is higher
    # than the expected (hence why it isn't squared or the abs())
    dW, dA, dB, db, dc = dtheta
    cost = @get_or_return(ctx, :sparsity_cost, nothing)
    target = @get(ctx, :sparsity_target,
                  throw(ArgumentError("If :sparsity_cost is used, :sparsity_target should also be defined")))
    curr_sparsity = mean(hid_means(rbm, X))
    penalty = cost * (curr_sparsity - target)
    axpy!(-penalty, dW, dW)
    axpy!(-penalty, dA, dA)
    axpy!(-penalty, dB, dB)
    axpy!(-penalty, db, db)
    axpy!(-penalty, dc, dc)
end


function update_weights!(crbm::ConditionalRBM, dtheta::Tuple, ctx::Dict)
    dW, dA, dB, db, dc = dtheta
    axpy!(1.0, dW, crbm.W)
    axpy!(1.0, dA, crbm.A)
    axpy!(1.0, dB, crbm.B)
    crbm.vbias += db
    crbm.hbias += dc
    # save previous dW
    dW_prev = @get_array(ctx, :dW_prev, size(dW), similar(dW))
    copy!(dW_prev, dW)
end


function update_classic!{T}(crbm::ConditionalRBM, X::Mat{T},
                            dtheta::Tuple, ctx::Dict)
    # apply gradient updaters. note, that updaters all have
    # the same signature and are thus composable
    grad_apply_learning_rate!(crbm, X, dtheta, ctx)
    grad_apply_momentum!(crbm, X, dtheta, ctx)
    grad_apply_weight_decay!(crbm, X, dtheta, ctx)
    grad_apply_sparsity!(crbm, X, dtheta, ctx)
    # add gradient to the weight matrix
    update_weights!(crbm, dtheta, ctx)
end


function free_energy{T}(crbm::ConditionalRBM, vis::Mat{T})
    vb = sum(vis .* crbm.dyn_vbias, 1)
    Wx_b_log = sum(log(1 + exp(crbm.W * vis .+ crbm.dyn_hbias)), 1)
    return - vb - Wx_b_log
end


function fit_batch!{T}(crbm::ConditionalRBM, X::Mat{T}, ctx = Dict())
    grad = @get_or_create(ctx, :gradient, gradient_classic)
    upd = @get_or_create(ctx, :update, update_classic!)
    curr, cond = split_vis(crbm, X)
    dynamic_biases!(crbm, cond)
    dtheta = grad(crbm, X, ctx)
    upd(crbm, X, dtheta, ctx)
    return crbm
end


function fit{T}(crbm::ConditionalRBM, X::Mat{T}, ctx = Dict{Any,Any}())
    @assert minimum(X) >= 0 && maximum(X) <= 1
    n_examples = size(X, 2)
    batch_size = @get(ctx, :batch_size, 100)
    n_batches = Int(ceil(n_examples / batch_size))
    n_epochs = @get(ctx, :n_epochs, 10)
    scorer = @get_or_create(ctx, :scorer, pseudo_likelihood)
    reporter = @get_or_create(ctx, :reporter, TextReporter())
    for epoch=1:n_epochs
        epoch_time = @elapsed begin
            for i=1:n_batches
                batch = X[:, ((i-1)*batch_size + 1):min(i*batch_size, end)]
                # batch = full(batch)
                fit_batch!(crbm, batch, ctx)
            end
        end
        curr, cond = split_vis(crbm, X)
        dynamic_biases!(crbm, cond)
        score = scorer(crbm, curr)
        report(reporter, crbm, epoch, epoch_time, score)
    end
    return crbm
end

fit{T}(crbm::ConditionalRBM, X::Mat{T}; opts...) = fit(crbm, X, Dict(opts))


function transform{T}(crbm::ConditionalRBM, X::Mat{T})
    curr, cond = split_vis(crbm, X)
    dynamic_biases!(crbm, cond)
    return hid_means(crbm, curr)
end


function generate{T}(crbm::ConditionalRBM, X::Mat{T}; n_gibbs=1)
    curr, cond = split_vis(crbm, X)
    dynamic_biases!(crbm, cond)
    return gibbs(crbm, curr; n_times=n_gibbs)[3]
end

generate{T}(crbm::ConditionalRBM, vis::Vec{T}; n_gibbs=1) =
    generate(crbm, reshape(vis, length(vis), 1); n_gibbs=n_gibbs)


function predict{T}(crbm::ConditionalRBM, cond::Mat{T}; n_gibbs=1)
    @assert size(cond, 1) == size(crbm.A, 2)

    curr = sub(cond, 1:length(crbm.vbias), :)
    vis = vcat(curr, cond)

    return generate(crbm, vis; n_gibbs=n_gibbs)
end

predict{T}(crbm::ConditionalRBM, cond::Vec{T}; n_gibbs=1) =
    predict(crbm, reshape(cond, length(cond), 1); n_gibbs=n_gibbs)


predict{T}(crbm::ConditionalRBM, vis::Vec{T}, cond::Vec{T}; n_gibbs=1) =
    predict(crbm, reshape(vis, length(vis), 1),
            reshape(cond, length(cond), 1); n_gibbs=n_gibbs)
