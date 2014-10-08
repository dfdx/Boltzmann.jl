
using Distributions
using Base.LinAlg.BLAS

abstract RBM

type BernoulliRBM <: RBM
    weights::Matrix{Float64}
    vbias::Vector{Float64}
    hbias::Vector{Float64}
    dW_prev::Matrix{Float64}
    momentum::Float64
    function BernoulliRBM(n_vis::Int, n_hid::Int; sigma=0.001, momentum=0.9)
        new(rand(Normal(0, sigma), (n_hid, n_vis)),
            zeros(n_vis), zeros(n_hid),
            zeros(n_hid, n_vis),
            momentum)
    end
    function Base.show(io::IO, rbm::BernoulliRBM)
        n_vis = size(rbm.vbias, 1)
        n_hid = size(rbm.hbias, 1)
        print("BernoulliRBM($n_vis, $n_hid)")
    end
end
    
    
type GRBM <: RBM
    weights::Matrix{Float64}
    vbias::Vector{Float64}
    hbias::Vector{Float64}
    dW_prev::Matrix{Float64}
    momentum::Float64
    function GRBM(n_vis::Int, n_hid::Int; sigma=0.001, momentum=0.9)
        new(rand(Normal(0, sigma), (n_hid, n_vis)),
            zeros(n_vis), zeros(n_hid),
            zeros(n_hid, n_vis),
            momentum)
    end
    function Base.show(io::IO, rbm::GRBM)
        n_vis = size(rbm.vbias, 1)
        n_hid = size(rbm.hbias, 1)
        print("GRBM($n_vis, $n_hid)")
    end
end




function logistic(x)
    return 1 ./ (1 + exp(-x))
end


function mean_hiddens{RBM <: RBM}(rbm::RBM, vis::Matrix{Float64})
    p = rbm.weights * vis .+ rbm.hbias
    return logistic(p)
end


function sample_hiddens{RBM <: RBM}(rbm::RBM, vis::Matrix{Float64})
    p = mean_hiddens(rbm, vis)
    return float(rand(size(p)) .< p)
end


function sample_visibles(rbm::BernoulliRBM, hid::Matrix{Float64})
    p = rbm.weights' * hid .+ rbm.vbias
    p = logistic(p)
    return float(rand(size(p)) .< p)
end


function sample_visibles(rbm::GRBM, hid::Matrix{Float64})
    mu = logistic(rbm.weights' * hid .+ rbm.vbias)
    sigma2 = 0.01                   # using fixed standard diviation
    samples = zeros(size(mu))
    for j=1:size(mu, 2), i=1:size(mu, 1)
        samples[i, j] = rand(Normal(mu[i, j], sigma2))
    end
    return samples
end


function gibbs{RBM <: RBM}(rbm::RBM, vis::Matrix{Float64}; n_times=1)
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


function free_energy{RBM <: RBM}(rbm::RBM, vis::Matrix{Float64})
    vb = sum(vis .* rbm.vbias, 1)
    Wx_b_log = sum(log(1 + exp(rbm.weights * vis .+ rbm.hbias)), 1)
    return - vb - Wx_b_log
end


function score_samples{RBM <: RBM}(rbm::RBM, vis::Matrix{Float64})
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


# TODO: remove
function mkAll(rows=100, cols=200)
    X = randn(rows, cols)
    X = X / (maximum(X) - minimum(X))
    X = X - minimum(X)
    rbm = GRBM(100, 50)
    return rbm, X
end
using PyPlot
ihist(X) = plt.hist(reshape(X, prod(size(X))), bins=100)

function update_weights!(rbm, h_pos, v_pos, h_neg, v_neg, lr, buf)
    dW = buf
    # dW = (h_pos * v_pos') - (h_neg * v_neg')
    gemm!('N', 'T', 1.0, h_neg, v_neg, 0.0, dW)
    gemm!('N', 'T', 1.0, h_pos, v_pos, -1.0, dW)
    # rbm.weights += lr * dW
    axpy!(lr, dW, rbm.weights)
    # rbm.weights += momentum * rbm.dW_prev
    axpy!(lr * rbm.momentum, rbm.dW_prev, rbm.weights)
    # save current dW
    copy!(rbm.dW_prev, dW)
end


function fit_batch!{RBM <: RBM}(rbm::RBM, vis::Matrix{Float64};
                    buf=None, lr=0.1, n_gibbs=1)
    buf = buf == None ? zeros(size(rbm.weights)) : buf 
    v_pos, h_pos, v_neg, h_neg = gibbs(rbm, vis, n_times=n_gibbs)
    lr = lr / size(v_pos, 1)
    update_weights!(rbm, h_pos, v_pos, h_neg, v_neg, lr, buf)
    rbm.hbias += vec(lr * (sum(h_pos, 2) - sum(h_neg, 2)))
    rbm.vbias += vec(lr * (sum(v_pos, 2) - sum(v_neg, 2)))
    return rbm
end


function fit{RBM <: RBM}(rbm::RBM, X::Matrix{Float64};
              lr=0.1, n_iter=10, batch_size=10, n_gibbs=1)
    @assert minimum(X) >= 0 && maximum(X) <= 1
    n_samples = size(X, 2)
    n_batches = int(ceil(n_samples / batch_size))
    w_buf = zeros(size(rbm.weights))
    for itr=1:n_iter
        tic()
        for i=1:n_batches
            batch = X[:, ((i-1)*batch_size + 1):min(i*batch_size, end)]
            fit_batch!(rbm, batch, buf=w_buf, n_gibbs=n_gibbs)
        end
        toc()
        @printf("Iteration #%s, pseudo-likelihood = %s\n",
                itr, mean(score_samples(rbm, X)))
    end
end


function transform{RBM <: RBM}(rbm::RBM, X::Matrix{Float64})
    return mean_hiddens(rbm, X)
end

