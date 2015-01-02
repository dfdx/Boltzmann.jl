
immutable DBN
    layers::Vector{(String, RBM)}
end


function Base.show(io::IO, dbn::DBN)
    layer_names = [getname(dbn, i) for i=1:length(dbn.layers)]
    layer_str = join(layer_names, ",")
    print(io, "DBN($layer_str)")
end

# DBN fields may change in the future, so it's worth to work through accessors
getname(dbn::DBN, k::Int) = dbn.layers[k][1]
getmodel(dbn::DBN, k::Int) = dbn.layers[k][2]
function getmodel(dbn::DBN, name::String)
    k = findfirst(p -> p[1] == name, dbn.layers)
    return (k != 0) ? dbn.layers[k][2] : error("No layer named '$name'")
end

# DBN may be thought of as a list of named RBMs, thus, we allow to access these RBMs
# with a short syntax like dbn[k] or dbn["layer_name"]
getindex(dbn::DBN, k::Int) = getmodel(dbn, k)
getindex(dbn::DBN, name::String) = getmodel(dbn, name)
Base.length(dbn::DBN) = length(dbn.layers)
Base.endof(dbn::DBN) = length(dbn)


function mh_at_layer(dbn::DBN, batch::Array{Float64, 2}, layer::Int)
    hiddens = Array(Array{Float64, 2}, layer)
    hiddens[1] = mean_hiddens(dbn[1], batch)
    for k=2:layer
        hiddens[k] = mean_hiddens(dbn[k], hiddens[k-1])
    end
    hiddens[end]
end


function transform(dbn::DBN, X::Mat{Float64})
    return mh_at_layer(dbn, X, length(dbn))
end


function fit(dbn::DBN, X::Mat{Float64}; lr=0.1, n_iter=10, batch_size=100, n_gibbs=1)
    @assert minimum(X) >= 0 && maximum(X) <= 1
    n_samples = size(X,2)
    n_batches = int(ceil(n_samples / batch_size))
    for k = 1:length(dbn.layers)
        w_buf = zeros(size(dbn[k].W))
        for itr=1:n_iter
            info("Layer $(k), iteration $(itr)")
            for i=1:n_batches
                batch = X[:, ((i-1)*batch_size + 1):min(i*batch_size, end)]
                input = k == 1 ? batch : mh_at_layer(dbn, batch, k-1)
                fit_batch!(dbn[k], input, buf=w_buf, n_gibbs=n_gibbs)
            end
        end
    end
end


