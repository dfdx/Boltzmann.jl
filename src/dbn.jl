# This is a DBN where the first layer is a either a Bernoulli or a 
# Gaussian-Bernoulli RBM and all the other layers are Bernoulli RBMs.
immutable DBN
  rbms::Vector{Union(BernoulliRBM, GRBM)}

  DBN(dims::Vector{Int}; vis_type=GRBM) = begin
  rbms = Array(Union(BernoulliRBM, GRBM), 0)
  if vis_type == GRBM
    push!(rbms, GRBM(dims[1],dims[2]))
  else
    if vis_type == BernoulliRBM
      push!(rbms, BernoulliRBM(dims[1],dims[2]))
    else 
      error("vis_type not supported")
    end
  end
  for k=2:length(dims)-1
    push!(rbms, BernoulliRBM(dims[k],dims[k+1]))
  end
  new(rbms)
end

end

function mh_at_layer(dbn::DBN, batch::Array{Float64, 2}, layer::Int)
  hiddens = Array(Array{Float64, 2}, layer)
  hiddens[1] = mean_hiddens(dbn.rbms[1], batch)
  for k=2:layer
    hiddens[k] = mean_hiddens(dbn.rbms[k], hiddens[k-1])
  end
  hiddens[end]
end

function fit(dbn::DBN, X::Mat{Float64}; lr=0.1, n_iter=10, batch_size=100, n_gibbs=1)
  n_samples = size(X,2)
  n_batches = int(ceil(n_samples / batch_size))
  for k = 1:length(dbn.rbms)
    w_buf = zeros(size(dbn.rbms[k].W))
    for itr=1:n_iter
      info("Layer $(k), iteration $(itr)")
      for i=1:n_batches
        batch = X[:, ((i-1)*batch_size + 1):min(i*batch_size, end)]
        input = k == 1 ? batch : mh_at_layer(dbn, batch, k)
        fit_batch!(dbn.rbms[k], input, buf=w_buf, n_gibbs=n_gibbs)
      end
    end
  end
end
