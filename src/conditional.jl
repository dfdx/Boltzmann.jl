
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


@runonce mutable struct ConditionalRBM{T,V,H} <: AbstractRBM{T,V,H}
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
        map(T, rand(Normal(0, sigma), n_hid, n_vis)),
        map(T, rand(Normal(0, sigma), n_vis, n_cond)),
        map(T, rand(Normal(0, sigma), n_hid, n_cond)),
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


function Base.show(io::IO, crbm::ConditionalRBM{T,V,H}) where {T,V,H}
    n_vis = size(crbm.vbias, 1)
    n_hid = size(crbm.hbias, 1)
    n_cond = size(crbm.A, 2)
    print(io, "ConditionalRBM{$T,$V,$H}($n_vis, $n_hid, $n_cond)")
end


function split_vis(crbm::ConditionalRBM, vis::Mat)
    vis_size = length(crbm.vbias)

    curr = vis[1:vis_size, :]
    cond = vis[(vis_size + 1):(vis_size + size(crbm.A, 2)), :]

    return curr, cond
end


function dynamic_biases!(crbm::ConditionalRBM, cond::Mat)
    crbm.dyn_vbias = crbm.A * cond .+ crbm.vbias
    crbm.dyn_hbias = crbm.B * cond .+ crbm.hbias
end


function hid_means(crbm::ConditionalRBM, vis::Mat)
    p = crbm.W * vis .+ crbm.dyn_hbias
    return logistic(p)
end


function vis_means(crbm::ConditionalRBM, hid::Mat)
    p = crbm.W' * hid .+ crbm.dyn_vbias
    return logistic(p)
end


function gradient_classic(crbm::ConditionalRBM{T}, X::Mat{T},
                       ctx::Dict) where T
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
    gemm!('N', 'T', T(1 / n_obs), v_neg, cond, T(0.0), dA)
    gemm!('N', 'T', T(1 / n_obs), v_pos, cond, T(-1.0), dA)


    # updating hid to hid matrix A
    dB = @get_array(ctx, :dB_buf, size(crbm.B), similar(crbm.B))
    gemm!('N', 'T', T(1 / n_obs), h_neg, cond, T(0.0), dB)
    gemm!('N', 'T', T(1 / n_obs), h_pos, cond, T(-1.0), dB)

    # gradient for vbias and hbias
    db = squeeze(sum(v_pos, 2) - sum(v_neg, 2), 2) ./ n_obs
    dc = squeeze(sum(h_pos, 2) - sum(h_neg, 2), 2) ./ n_obs
    return dW, dA, dB, db, dc
end


function grad_apply_learning_rate!(crbm::ConditionalRBM{T},
                                   X::Mat{T},
                                   dtheta::Tuple, ctx::Dict) where T
    dW, dA, dB, db, dc = dtheta
    lr = T(@get(ctx, :lr, 0.1))
    # same as: dW *= lr
    scal!(length(dW), lr, dW, 1)
    scal!(length(dA), lr, dA, 1)
    scal!(length(dB), lr, dB, 1)
    scal!(length(db), lr, db, 1)
    scal!(length(dc), lr, dc, 1)
end


function grad_apply_momentum!(crbm::ConditionalRBM{T}, X::Mat{T},
                              dtheta::Tuple, ctx::Dict) where T
    dW, dA, dB, db, dc = dtheta
    momentum = @get(ctx, :momentum, 0.9)
    dW_prev = @get_array(ctx, :dW_prev, size(dW), zeros(T, size(dW)))
    # same as: dW += momentum * dW_prev
    axpy!(momentum, dW_prev, dW)
end


function grad_apply_weight_decay!(rbm::ConditionalRBM,
                                     X::Mat,
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


function grad_apply_sparsity!(rbm::ConditionalRBM{T}, X::Mat,
                              dtheta::Tuple, ctx::Dict) where T
    # The sparsity constraint should only drive the weights
    # down when the mean activation of hidden units is higher
    # than the expected (hence why it isn't squared or the abs())
    dW, dA, dB, db, dc = dtheta
    cost = @get_or_return(ctx, :sparsity_cost, nothing)
    vis, cond = split_vis(rbm, X)
    target = @get(ctx, :sparsity_target,
                  throw(ArgumentError("If :sparsity_cost is used, :sparsity_target should also be defined")))
    curr_sparsity = mean(hid_means(rbm, vis))
    penalty = T(cost * (curr_sparsity - target))
    add!(dW, -penalty)
    add!(dA, -penalty)
    add!(dB, -penalty)
    add!(db, -penalty)
    add!(dc, -penalty)
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


function update_classic!(crbm::ConditionalRBM, X::Mat,
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


function free_energy(crbm::ConditionalRBM, vis::Mat)
    vb = sum(vis .* crbm.dyn_vbias, 1)
    Wx_b_log = sum(log.(1 + exp.(crbm.W * vis .+ crbm.dyn_hbias)), 1)
    result = - vb - Wx_b_log
    tofinite!(result)

    return result
end


function fit_batch!(crbm::ConditionalRBM, X::Mat, ctx = Dict())
    grad = @get_or_create(ctx, :gradient, gradient_classic)
    upd = @get_or_create(ctx, :update, update_classic!)
    curr, cond = split_vis(crbm, X)
    dynamic_biases!(crbm, cond)
    dtheta = grad(crbm, X, ctx)
    upd(crbm, X, dtheta, ctx)
    return crbm
end


function fit(crbm::ConditionalRBM{T}, X::Mat, opts::Dict{Any,Any}) where T
    @assert minimum(X) >= 0 && maximum(X) <= 1
    ctx = copy(opts)
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
                batch = ensure_type(T, batch)
                fit_batch!(crbm, batch, ctx)
            end
        end
        curr, cond = split_vis(crbm, X)
        dynamic_biases!(crbm, cond)

        # We convert to full, to avoid changing the the n_obs if
        # X is a sparse matrix
        score = scorer(crbm, full(curr))
        report(reporter, crbm, epoch, epoch_time, score)
    end

    return crbm
end

fit(crbm::ConditionalRBM{T}, X::Mat; opts...) where {T} = fit(crbm, X, Dict{Any,Any}(opts))


function transform(crbm::ConditionalRBM{T}, X::Mat) where T
    curr, cond = split_vis(crbm, ensure_type(T, X))
    dynamic_biases!(crbm, cond)
    return hid_means(crbm, curr)
end


function generate(crbm::ConditionalRBM{T}, X::Mat; n_gibbs=1) where T
    curr, cond = split_vis(crbm, ensure_type(T, X))
    dynamic_biases!(crbm, cond)
    return gibbs(crbm, curr; n_times=n_gibbs)[3]
end

generate(crbm::ConditionalRBM{T}, vis::Vec; n_gibbs=1) where {T} =
    generate(crbm, reshape(ensure_type(T, vis), length(vis), 1); n_gibbs=n_gibbs)


function predict(crbm::ConditionalRBM{T}, cond::Mat; n_gibbs=1) where T
    cond = ensure_type(T, cond)
    @assert size(cond, 1) == size(crbm.A, 2)

    curr = view(cond, 1:length(crbm.vbias), :)
    vis = vcat(curr, cond)

    return generate(crbm, vis; n_gibbs=n_gibbs)
end

predict(crbm::ConditionalRBM{T}, cond::Vec; n_gibbs=1) where {T} =
    predict(crbm, reshape(ensure_type(T, cond), length(cond), 1); n_gibbs=n_gibbs)


predict(crbm::ConditionalRBM{T}, vis::Vec, cond::Vec{T}; n_gibbs=1) where {T} =
    predict(crbm, reshape(vis, length(vis), 1),
            reshape(ensure_type(T, cond), length(cond), 1); n_gibbs=n_gibbs)
