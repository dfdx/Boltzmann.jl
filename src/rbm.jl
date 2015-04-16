
using Distributions
using Base.LinAlg.BLAS

abstract RBM

typealias Mat{T} AbstractArray{T, 2}
typealias Vec{T} AbstractArray{T, 1}

"""
The BernoulliRBM is the RBM
that necessarily takes as vectorized
input binary
units, represented as column vectors
or row vectors
"""
type BernoulliRBM <: RBM
    # The matrix of weights for the RBM
    W::Matrix{Float64}
    # The bias unit for the visible units
    vbias::Vector{Float64}
    # The bias unit for the hidden units
    hbias::Vector{Float64}
    # The difference in the previous matrix
    dW_prev::Matrix{Float64}
    # We'll learn what this is
    persistent_chain::Matrix{Float64}
    # The momentum for the RBM
    momentum::Float64
    # The pseudolikelihood arrived at at at the last training batch
    pseudolikelihood::Float64
    # Inner constructor, intializes the RBM to 
    function BernoulliRBM(n_vis::Int, n_hid::Int; sigma=0.001, momentum=0.9)
        new(
            rand(Normal(0, sigma), (n_hid, n_vis)), # Initlialize the weight matrix to be 
                                                    # randomly distributed about a mean of 0
                                                    # and a std deviation of 0.001
            zeros(n_vis), # initial vbias
            zeros(n_hid), # initial hbias        
            zeros(n_hid, n_vis), # difference matrix 
            Array(Float64, 0, 0), # persistent chain
            momentum,
            0.0) #initial pseudolikelihood
    end
    function Base.show(io::IO, rbm::BernoulliRBM)
        n_vis = size(rbm.vbias, 1)
        n_hid = size(rbm.hbias, 1)
        print(io, "BernoulliRBM($n_vis, $n_hid)")
    end
end


# The Gaussian RBM allows you to use non-binary units, 
# and instead uses a Gaussian squashing function to
# turn your units into effectively binary units
# Being that these are both actually the same type, 
# it may be better to find some other way of differentiating
# GRBMs frmo RBMs. Perhaps with a type alias.
type GRBM <: RBM
    W::Mat{Float64}
    vbias::Vec{Float64}
    hbias::Vec{Float64}
    dW_prev::Mat{Float64}
    persistent_chain::Matrix{Float64}
    momentum::Float64
    pseudolikelihood::Float64
    function GRBM(n_vis::Int, n_hid::Int; sigma=0.001, momentum=0.9)
        new(rand(Normal(0, sigma), (n_hid, n_vis)),
            zeros(n_vis), zeros(n_hid),
            zeros(n_hid, n_vis),
            Array(Float64, 0, 0),
            momentum,
            0.0)
    end
    function Base.show(io::IO, rbm::GRBM)
        n_vis = size(rbm.vbias, 1)
        n_hid = size(rbm.hbias, 1)
        print(io, "GRBM($n_vis, $n_hid)")
    end
end


function logistic(x)
  """The logistic function. Google it!"""
    return 1 ./ (1 + exp(-x))
end


function mean_hiddens(rbm::RBM, vis::Mat{Float64})
  """Calculate the activations of the hidden units,
  and return the probabiltiies of the hidden units turning on.
  Note that we are adding the hidden bias unit to the visisble 
  units"""
    p = gemv("N", 1, rbm.W, vis .+ rbm.hbias) 
    return logistic(p)
end


function sample_hiddens(rbm::RBM, vis::Mat{Float64})
  """After finding the probabilities of the hidden units
  activating, actually see which hidden units activate through
  a probabilistic sampling"""
    p = mean_hiddens(rbm, vis)
    return float(rand(size(p)) .< p)
end


function sample_visibles(rbm::BernoulliRBM, hid::Mat{Float64})
    """Calculate the activations and see which visible units
    activate. Essentially the same as `sample_hiddens` except 
    now we're finding the probabilities that a visible unit 
    would go off for a given hidden matrix"""
    p = gemv("T", 1, rbm.W, hid .+ rbm.vbias) 
    p = logistic(p)
    return float(rand(size(p)) .< p)
end


function sample_visibles(rbm::GRBM, hid::Mat{Float64})
    """Same as sample visibles for the BernoulliRBM,
       but using the Gaussian function rather than the 
       logistic function"""
    mu = logistic(gemv("T", 1, rbm.W, hid .+ rbm.vbias)
    sigma2 = 0.01                   # using fixed standard diviation
    samples = zeros(size(mu))
    for j=1:size(mu, 2), i=1:size(mu, 1)
        samples[i, j] = rand(Normal(mu[i, j], sigma2))
    end
    return samples
end


function gibbs(rbm::RBM, vis::Mat{Float64}; n_times=1)
    """
    Perform a gibbs sampling:
    v_pos is part of the positive phase of the Gradient, taken as the given input matrix

    h_pos is part of the positive phase of the Gradient, given by sampling the hidden unit
    activations using v_pos

    v_neg is part of the negative phase of the Gradient, taken by using the samples obtained
    for v_pos and h_pos, and propagating that back through and sampling the visible
    layer with the h_pos, the activations from the visible layer.

    h_neg is part of the negative phase of the Gradient, found by taking v_neg and sampling
    the hidden units using v_neg. 
    
    We can perform gibb's sampling n times and return a sufficiently sampled random number. Not the point
    
    The question you're asking is, why does Gibbs sampling work?
    
    Because the probabilities of each of the layer is conditionally independent, we can 'walk the markov chain' of
    sampling by sampling the visible units, and then the hidden units, and then the visible and hidden units again.

    Why are we doing this? The second term of the negative log likelihood of the gradient of
    the free energy function we constructed above involves taking the expectation value of a 
    function analytically related to the free energy function (it's derivative 
    with respect to the model parameters (basically the weights
    and bias units) but over all the possible configurations
    of the input vector x. This is very computationally expensive,
    so instead, if we use a Gibb's sampling,
    we can instead arrive at an approximation 
    to the expectation value of the gradient, by
    using a set of samples. These samples are referred
    to as negative particles. 
    """
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


function free_energy(rbm::RBM, vis::Mat{Float64})
    """
    The free energy of our RBM is essentially the 'cost function' we want to minimize,
    The energy of a pair of boolean vectors v (for visible) and h (for hidden),
    E(v,h) = -vbias'*v - hbias'*h - v'*rbm.W*h

    The free energy of a given visible vector, using binary units, actually becomes quite simple,
    since the probability of an activation goes down, and thus the negative log likelihood becomes
    easier to understand. 
    F(v) = -vbias*v - \sum_i log(1+ exp(hbias + W*v) )
    """
    # First we need to get a 
    vb = sum(vis .* rbm.vbias, 1)
    Wx_b_log = sum(log(1 + exp(rbm.W * vis .+ rbm.hbias)), 1)
    return - vb - Wx_b_log
end


function score_samples(rbm::RBM, vis::Mat{Float64}; sample_size=10000)
  """Score samples works to see how well your fitted weight matrix
  works to minimize the free energy of your RBM and visual matrix system"""
    if issparse(vis)
        # sparse matrices may be infeasible for this operation
        # so using only little sample
        # We need somewhere to expose the sample size in the API
        cols = sample(1:size(vis, 2), sample_size)
        vis = full(vis[:, cols])
    end
    n_feat, n_samples = size(vis)
    vis_corrupted = copy(vis)
    # Create a random vector of length
    # equal to the number of features, and
    # with a maximum number equal to the number
    # of saamples we're considering
    idxs = rand(1:n_feat, n_samples)
    # For each random feature, and for each sample
    # we'll create a corrupted matrix. By this, we 
    # will create a matrix with much higher entropy
    # than the visual matrix we are comparing against.
    # We do this by taking random elements and switching 
    # their 'bits', so that the new samples are randomly
    # distributed (ideally compared to the initial 
    # set of samples.
    for (i, j) in zip(idxs, 1:n_samples)
        vis_corrupted[i, j] = 1 - vis_corrupted[i, j]
    end
    # Then we calculate the free energy for 
    # both the corrputed visual matrix and the regular 
    # visual matrix
    fe = free_energy(rbm, vis)
    fe_corrupted = free_energy(rbm, vis_corrupted)
    # Our pseudolikelihood then is the number of features
    # times the log likelihood of the difference between the
    # corrupted measurement and the actual measurement. In other
    # words, the pseudo likelihood is a measure of the distance
    # in the probability between the system we have fit to 
    # and a system somewhat similar to ours but in actuality
    # very different in terms of having random activations. 
    #TODO: If we decide to use constrastive divergence, 
    # we should instead use the reconstruction cross-entropy
    # rather than the pseudo likelihood
    return n_feat * log(logistic(fe_corrupted - fe))
end


function update_weights!(rbm, h_pos, v_pos, h_neg, v_neg, lr, buf)
  """ 
  buf is probably zero. First, we assign it to the gradient that we calculate. The
  gradient is the dot product of the positive activations minus the dot product of the 
  negative activations.
  Then we assign the weight matrices to the learning rate times the gradient
  finally, we descend on the gradient by taking our current gradient, multiplying it 
  by the prior gradient, and multiplying by the momentum
  We then keep track of the prior gradient. 
  """
    dW = buf
    # dW = (h_pos * v_pos') - (h_neg * v_neg')
    gemm!('N', 'T', 1.0, h_neg, v_neg, 0.0, dW)
    gemm!('N', 'T', 1.0, h_pos, v_pos, -1.0, dW)
    # rbm.W += lr * dW
    axpy!(lr, dW, rbm.W)
    # rbm.W += rbm.momentum * rbm.dW_prev
    axpy!(lr * rbm.momentum, rbm.dW_prev, rbm.W)
    # save current dW
    copy!(rbm.dW_prev, dW)
end


function contdiv(rbm::RBM, vis::Mat{Float64}, n_gibbs::Int)
  """contdiv performs gibbs sampling and traverses the markov
  chain of virtual particles n times"""
    v_pos, h_pos, v_neg, h_neg = gibbs(rbm, vis, n_times=n_gibbs)
    return v_pos, h_pos, v_neg, h_neg
end


function persistent_contdiv(rbm::RBM, vis::Mat{Float64}, n_gibbs::Int)
  """persistent_contdiv performs gibbs sampling, traversing the markov
  chain n times. Since this is done multiple times per epoch, and also
  per batch, n_gibbs can probably stay at one or a low number, since the 
  markov chain being persistent will result in further traversal as more 
  batches are fitted"""
    if size(rbm.persistent_chain) != size(vis)
        # persistent_chain not initialized or batch size changed, re-initialize
        rbm.persistent_chain = vis
    end
    # take positive samples from real data
    v_pos, h_pos, _, _ = gibbs(rbm, vis)
    # take negative samples from "fantasy particles"
    rbm.persistent_chain, _, v_neg, h_neg = gibbs(rbm, vis, n_times=n_gibbs)
    return v_pos, h_pos, v_neg, h_neg
end


function fit_batch!(rbm::RBM, vis::Mat{Float64};
                    persistent=true, buf=None, lr=0.1, n_gibbs=1)
    """fit_batch! creates our sampler (dependent on whether or not we're
    choosing to use persistent chain sampling or not) and the sampler then
    facilitates the ggibbs sampling. After the sampling is done, the weight matrices are 
    updated and the bias vectors are updated as well"""
    # We pass in a buffer. Not sure why there's logic for checking
    # buffer if we only ever call this function after coming from 
    # fit, but whatever
    buf = buf == None ? zeros(size(rbm.W)) : buf
    # v_pos, h_pos, v_neg, h_neg = gibbs(rbm, vis, n_times=n_gibbs)
    # Let's figure out what the hell persistent means
    sampler = persistent ? persistent_contdiv : contdiv
    # check out sampler
    v_pos, h_pos, v_neg, h_neg = sampler(rbm, vis, n_gibbs)
    lr = lr / size(v_pos, 1)
    update_weights!(rbm, h_pos, v_pos, h_neg, v_neg, lr, buf)
    rbm.hbias += vec(lr * (sum(h_pos, 2) - sum(h_neg, 2)))
    rbm.vbias += vec(lr * (sum(v_pos, 2) - sum(v_neg, 2)))
    return rbm
end


function fit(rbm::RBM, X::Mat{Float64};
             persistent=true, lr=0.1, n_iter=10, batch_size=100, n_gibbs=1)
    """Work to minimize the free energy of the RBM
    given an input matrix
    
    persistent: Whether or not we will use a persistent chain. I do not know if it is
        advised in this library not to use a persistent chain, as it will be much harder to find
        convergence. A persistent chain involves keeping the visual matrix used for fitting as the 
        source of positive examples for all of the gibb's sampling steps in generating the negative
        particles. If persistent is turned off, then every gibb's sampling set essentially draws
        from a new markov chain when sampling the hidden units, rather than continuing to 
        draw from the visual matrix you gave. The tradeoff as I see it is that in exploring phase space
        for training, keeping a persistent chain ensures that you have a higher chance of eventually finding
        equilibrium, while not having a persistent change allows you to continue to sample different 
        starting points in your search through phase space for a minima, as such allowing you to 
        potentially find a minima faster but at a high risk of undoing that work during a subsequent sampling step.

      n_gibbs: n_gibbs determines the number of times to perform the gibb's sampling.
        In other words, the number n_gibbs is set to is the number of times the markov chain 
        generated for the negative particles is walked. I think I would recommend n_gibbs to be set to something
        other than on if persistent is not true """
    # First, we ensure that we have an actual binary matrix
    @assert minimum(X) >= 0 && maximum(X) <= 1
    # Next, we determine the number of samples. Recall that
    # for some reason (probably optimization), our matrices are
    # columnwise, so the number of samples we'll get is size(X, 2),
    # or the number of columns in the matrix
    n_samples = size(X, 2)
    # We find the number of batches we'll use 
    n_batches = int(ceil(n_samples / batch_size))
    # 
    w_buf = zeros(size(rbm.W))
    # For each epoch
    for itr=1:n_iter
        tic()
        for i=1:n_batches
            # println("fitting $(i)th batch")
            # Pull the first $batch_size columns 
            batch = X[:, ((i-1)*batch_size + 1):min(i*batch_size, end)]
            # turn the matrix into a full matrix if it was a sparse matrix
            batch = full(batch)
            fit_batch!(rbm, batch, persistent=persistent,
                       buf=w_buf, n_gibbs=n_gibbs)
        end
        toc()
        # After fitting, see how well your new weight matrix
        # minimizes your free energy
        pseudo_likelihood = mean(score_samples(rbm, X))
        rbm.pseudolikelihood = pseudo_likelihood
        info("Iteration #$itr, pseudo-likelihood = $pseudo_likelihood")
    end
    return rbm
end


function transform(rbm::RBM, X::Mat{Float64})
    """Transform just returns the activations
    of the hidden units on the given matrix X.
    This is what you want out of your RBM eventually
    probably"""
    return mean_hiddens(rbm, X)
end

function generate(rbm::RBM, vis::Vec{Float64}; n_gibbs=1)
  """
  generate performs gibbs sampling steps on your vector. 
  Increasing the n_gibbs has a higher chance of sampling a useful
  portion of your markov chain"""
    return gibbs(rbm, reshape(vis, length(vis), 1); n_times=n_gibbs)[3]
end

function generate(rbm::RBM, X::Mat{Float64}; n_gibbs=1)
  """generate performs gibbs sampling steps on your matrix. 
  Increasing the n_gibbs has a higher chance of sampling a useful
  portion of your markov chain"""
    return gibbs(rbm, X; n_times=n_gibbs)[3]
end


function components(rbm::RBM; transpose=true)
    return if transpose rbm.W' else rbm.W end
end
# synonym
features(rbm::RBM; transpose=true) = components(rbm, transpose)
