
## Julia implementation of a ConditionalRBM from Graham Taylor's PhD Thesis

## Links:
##     Thesis - http://www.cs.nyu.edu/~gwtaylor/thesis/Taylor_Graham_W_200911_PhD_thesis.pdf
##     FCRBM - http://www.cs.toronto.edu/~fritz/absps/fcrbm_icml.pdf

import StatsBase: predict


@runonce type ConditionalRBM{T,V,H} <: AbstractRBM
    W::Matrix{T}  # standard weights
    A::Matrix{T}  # autoregressive params (vis to vis)
    B::Matrix{T}  # hidden params(vis to hid)
    vbias::Vector{T}
    hbias::Vector{T}
    dyn_vbias::Array{T}
    dyn_hbias::Array{T}
    steps::Int
    # dW_prev::Matrix{T}
    # persistent_chain::Matrix{T}
    # momentum::Float64
end


function ConditionalRBM(T::Type, V::Type, H::Type,
                        n_vis::Int, n_hid::Int, steps::Int; sigma=0.01)
    ConditionalRBM{T,V,H}(
        map(T, rand(Normal(0, sigma), (n_hid, n_vis))),      
        map(T, rand(Normal(0, sigma), (n_vis, n_vis * steps))),
        map(T, rand(Normal(0, sigma), (n_hid, n_vis * steps))),
        zeros(T, n_vis),
        zeros(T, n_hid),
        zeros(T, n_vis),
        zeros(T, n_hid),
        steps)
end

function Base.show{T,V,H}(io::IO, crbm::ConditionalRBM{T,V,H})
    n_vis = size(crbm.vbias, 1)
    n_hid = size(crbm.hbias, 1)
    steps = crbm.steps
    print(io, "ConditionalRBM{$T,$V,$H}($n_vis, $n_hid, $steps)")
end

function split_vis{T}(crbm::ConditionalRBM, vis::Mat{T})
    curr_end = length(crbm.vbias)
    hist_start = curr_end + 1
    hist_end = curr_end + curr_end * crbm.steps

    curr = sub(vis, 1:curr_end, :)
    hist = sub(vis, hist_start:hist_end, :)
    return curr, hist
end

function dynamic_biases!{T}(crbm::ConditionalRBM, history::Mat{T})
    crbm.dyn_vbias = crbm.A * history .+ crbm.vbias
    crbm.dyn_hbias = crbm.B * history .+ crbm.hbias
end

function hid_means{T}(crbm::ConditionalRBM, vis::Mat{T})
    p = crbm.W * vis .+ crbm.dyn_hbias
    return logistic(p)
end

function vis_means{T}(crbm::ConditionalRBM, hid::Mat{T})
    p = crbm.W' * hid .+ crbm.dyn_vbias
    return logistic(p)
end


function gradient_classic{T}(crbm::ConditionalRBM, vis::Mat{T}, config::Dict)
    sampler = @get_or_create(config, :sampler, persistent_contdiv)
    v_pos, h_pos, v_neg, h_neg = sampler(rbm, vis, config)
    dW = @get_array(config, :dW_buf, size(rbm.W), similar(rbm.W))
    n_obs = size(vis, 2)
    # same as: dW = (h_pos * v_pos') - (h_neg * v_neg')
    gemm!('N', 'T', T(1 / n_obs), h_neg, v_neg, T(0.0), dW)
    gemm!('N', 'T', T(1 / n_obs), h_pos, v_pos, T(-1.0), dW)
    # gradient for vbias and hbias
    db = squeeze(sum(v_pos, 2) - sum(v_neg, 2), 2) ./ n_obs
    dc = squeeze(sum(h_pos, 2) - sum(h_neg, 2), 2) ./ n_obs
    return dW, db, dc
end


# No momentum for params
function update_weights!(crbm::ConditionalRBM, h_pos, v_pos, h_neg, v_neg, hist, lr, buf)
    # Normal W weight update
    dW = buf[1]

    # dW = (h_pos * v_pos') - (h_neg * v_neg')
    gemm!('N', 'T', lr, h_neg, v_neg, 0.0, dW)
    gemm!('N', 'T', lr, h_pos, v_pos, -1.0, dW)
    # crbm.dW += crbm.momentum * crbm.dW_prev
    axpy!(crbm.momentum, crbm.dW_prev, dW)
    # rbm.W += lr * dW
    axpy!(1.0, dW, crbm.W)
    # save current dW
    copy!(crbm.dW_prev, dW)

    # Update A (history -> vis) weights
    dA = buf[2]

    # dW = (v_pos * hist') - (v_neg * hist')
    gemm!('N', 'T', lr, v_neg, hist, 0.0, dA)
    gemm!('N', 'T', lr, v_pos, hist, -1.0, dA)
    # rbm.A += lr * dW
    axpy!(1.0, dA, crbm.A)

    # Update B (history -> hid) weights
    dB = buf[3]

    # dW = (h_pos * hist') - (h_neg * hist')
    gemm!('N', 'T', lr, h_neg, hist, 0.0, dB)
    gemm!('N', 'T', lr, h_pos, hist, -1.0, dB)
    # rbm.B += lr * dW
    axpy!(1.0, dB, crbm.B)
end

function free_energy{T}(crbm::ConditionalRBM, vis::Mat{T})
    vb = sum(vis .* crbm.dyn_vbias, 1)
    Wx_b_log = sum(log(1 + exp(crbm.W * vis .+ crbm.dyn_hbias)), 1)
    return - vb - Wx_b_log
end

function fit_batch!{T}(crbm::ConditionalRBM, vis::Mat{T};
                    persistent=true, buf=nothing, lr=0.1, n_gibbs=1)
    buf = buf == nothing ? (zeros(size(crbm.W)), zeros(size(crbm.A)), zeros(size(crbm.B))) : buf
    curr, hist = split_vis(crbm, vis)
    dynamic_biases!(crbm, hist)
    # v_pos, h_pos, v_neg, h_neg = gibbs(rbm, vis, n_times=n_gibbs)
    sampler = persistent ? persistent_contdiv : contdiv
    v_pos, h_pos, v_neg, h_neg = sampler(crbm, curr, n_gibbs)
    lr=lr/size(v_pos,2)
    update_weights!(crbm, h_pos, v_pos, h_neg, v_neg, hist, lr, buf)
    crbm.hbias += vec(lr * (sum(h_pos, 2) - sum(h_neg, 2)))
    crbm.vbias += vec(lr * (sum(v_pos, 2) - sum(v_neg, 2)))
    return crbm
end

function fit{T}(crbm::ConditionalRBM, X::Mat{T};
             persistent=true, lr=0.1, n_iter=10, batch_size=100, n_gibbs=1)
    @assert minimum(X) >= 0 && maximum(X) <= 1
    n_samples = size(X, 2)
    n_batches = Int(ceil(n_samples / batch_size))
    buffers = (zeros(size(crbm.W)), zeros(size(crbm.A)), zeros(size(crbm.B)))
    for itr=1:n_iter
        tic()
        for i=1:n_batches
            # println("fitting $(i)th batch")
            batch = X[:, ((i-1)*batch_size + 1):min(i*batch_size, end)]
            batch = full(batch)
            fit_batch!(crbm, batch, persistent=persistent,
                       buf=buffers, n_gibbs=n_gibbs)
        end
        toc()
        curr, hist = split_vis(crbm, X)
        dynamic_biases!(crbm, hist)
        pseudo_likelihood = mean(score_samples(crbm, curr))
        info("Iteration #$itr, pseudo-likelihood = $pseudo_likelihood")
    end
    return crbm
end

function transform{T}(crbm::ConditionalRBM, X::Mat{T})
    curr, hist = split_vis(crbm, X)
    dynamic_biases!(crbm, hist)
    return hid_means(crbm, curr)
end


function generate{T}(crbm::ConditionalRBM, X::Mat{T}; n_gibbs=1)
    curr, hist = split_vis(crbm, X)
    dynamic_biases!(crbm, hist)
    return gibbs(crbm, curr; n_times=n_gibbs)[3]
end

generate{T}(crbm::ConditionalRBM, vis::Vec{T}; n_gibbs=1) = generate(
    crbm, reshape(vis, length(vis), 1); n_gibbs=n_gibbs
)

function predict{T}(crbm::ConditionalRBM, history::Mat{T}; n_gibbs=1)
    @assert size(history, 1) == size(crbm.A, 2)

    curr = sub(history, 1:length(crbm.vbias), :)
    vis = vcat(curr, history)

    return generate(crbm, vis; n_gibbs=n_gibbs)
end

predict{T}(crbm::ConditionalRBM, history::Vec{T}; n_gibbs=1) = predict(
    crbm, reshape(history, length(history), 1); n_gibbs=n_gibbs
)

predict{T}(crbm::ConditionalRBM, vis::Mat{T}, hist::Mat{T}; n_gibbs=1) = generate(
    crbm, vcat(vis, hist); n_gibbs=n_gibbs
)

predict{T}(crbm::ConditionalRBM, vis::Vec{T}, hist::Vec{T}; n_gibbs=1) = predict(
    crbm, reshape(vis, length(vis), 1), reshape(hist, length(hist), 1); n_gibbs=n_gibbs
)
