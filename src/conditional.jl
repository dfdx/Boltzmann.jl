"""
Julia implementation of a ConditionalRBM from Graham Taylor's PhD Thesis

Links:
    Thesis - http://www.cs.nyu.edu/~gwtaylor/thesis/Taylor_Graham_W_200911_PhD_thesis.pdf
    FCRBM - http://www.cs.toronto.edu/~fritz/absps/fcrbm_icml.pdf
"""
import StatsBase: predict


@runonce type ConditionalRBM{V,H} <: AbstractRBM{V,H}
    W::Matrix{Float64}  # standard weights
    A::Matrix{Float64}  # autoregressive params (vis to vis)
    B::Matrix{Float64}  # hidden params(vis to hid)
    dyn_vbias::Array{Float64}
    dyn_hbias::Array{Float64}
    vbias::Vector{Float64}
    hbias::Vector{Float64}
    steps::Int
    dW_prev::Matrix{Float64}
    persistent_chain::Matrix{Float64}
    momentum::Float64
end

function ConditionalRBM(V::Type, H::Type, n_vis::Int, n_hid::Int;
        sigma=0.01, momentum=0.9, steps=5)

    ConditionalRBM{V,H}(
        rand(Normal(0, sigma), (n_hid, n_vis)),
        rand(Normal(0, sigma), (n_vis, n_vis * steps)),
        rand(Normal(0, sigma), (n_hid, n_vis * steps)),
        zeros(n_vis),
        zeros(n_hid),
        zeros(n_vis),
        zeros(n_hid),
        steps,
        zeros(n_hid, n_vis),
        Array(Float64, 0, 0),
        momentum,
    )
end

function Base.show{V,H}(io::IO, rbm::RBM{V,H})
    n_vis = size(rbm.vbias, 1)
    n_hid = size(rbm.hbias, 1)
    print(io, "RBM{$V,$H}($n_vis, $n_hid)")
end

function split_vis(crbm::ConditionalRBM, vis::Mat{Float64})
    curr_end = length(crbm.vbias)
    hist_start = curr_end + 1
    hist_end = curr_end + curr_end * crbm.steps

    curr = sub(vis, 1:curr_end, :)
    hist = sub(vis, hist_start:hist_end, :)
    return curr, hist
end

function dynamic_biases!(crbm::ConditionalRBM, history::Mat{Float64})
    crbm.dyn_vbias = crbm.A * history .+ crbm.vbias
    crbm.dyn_hbias = crbm.B * history .+ crbm.hbias
end

function hid_means(crbm::ConditionalRBM, vis::Mat{Float64})
    p = crbm.W * vis .+ crbm.dyn_hbias
    return logistic(p)
end

function vis_means(crbm::ConditionalRBM, hid::Mat{Float64})
    p = crbm.W' * hid .+ crbm.dyn_vbias
    return logistic(p)
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

function free_energy(crbm::ConditionalRBM, vis::Mat{Float64})
    vb = sum(vis .* crbm.dyn_vbias, 1)
    Wx_b_log = sum(log(1 + exp(crbm.W * vis .+ crbm.dyn_hbias)), 1)
    return - vb - Wx_b_log
end

function fit_batch!(crbm::ConditionalRBM, vis::Mat{Float64};
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

function fit(crbm::ConditionalRBM, X::Mat{Float64};
             persistent=true, lr=0.1, n_iter=10, batch_size=100, n_gibbs=1)
    @assert minimum(X) >= 0 && maximum(X) <= 1
    n_samples = size(X, 2)
    n_batches = @compat Int(ceil(n_samples / batch_size))
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

function transform(crbm::ConditionalRBM, X::Mat{Float64})
    curr, hist = split_vis(crbm, X)
    dynamic_biases!(crbm, hist)
    return hid_means(crbm, curr)
end


function generate(crbm::ConditionalRBM, X::Mat{Float64}; n_gibbs=1)
    curr, hist = split_vis(crbm, X)
    dynamic_biases!(crbm, hist)
    return gibbs(crbm, curr; n_times=n_gibbs)[3]
end

generate(crbm::ConditionalRBM, vis::Vec{Float64}; n_gibbs=1) = generate(
    crbm, reshape(vis, length(vis), 1); n_gibbs=n_gibbs
)

function predict(crbm::ConditionalRBM, history::Mat{Float64}; n_gibbs=1)
    @assert size(history, 1) == size(crbm.A, 2)

    curr = sub(history, 1:length(crbm.vbias), :)
    vis = vcat(curr, history)

    return generate(crbm, vis; n_gibbs=n_gibbs)
end

predict(crbm::ConditionalRBM, history::Vec{Float64}; n_gibbs=1) = predict(
    crbm, reshape(history, length(history), 1); n_gibbs=n_gibbs
)

predict(crbm::ConditionalRBM, vis::Mat{Float64}, hist::Mat{Float64}; n_gibbs=1) = generate(
    crbm, vcat(vis, hist); n_gibbs=n_gibbs
)

predict(crbm::ConditionalRBM, vis::Vec{Float64}, hist::Vec{Float64}; n_gibbs=1) = predict(
    crbm, reshape(vis, length(vis), 1), reshape(hist, length(hist), 1); n_gibbs=n_gibbs
)
