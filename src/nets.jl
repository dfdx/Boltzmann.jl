
# TODO: add support for custom T

import Base.getindex

abstract type Net end

struct DBN <: Net
    layers::Vector{RBM}
    layernames::Vector{AbstractString}
end

DBN(namedlayers::Vector{LT}) where {LT<:Tuple{AbstractString,RBM}} =
    DBN(map(p -> p[2], namedlayers), map(p -> p[1], namedlayers))
    

struct DAE <: Net
    layers::Vector{RBM}
    layernames::Vector{AbstractString}
end


function Base.show(io::IO, net::Net)
    nettype = string(typeof(net))
    layer_str = join(net.layernames, ",")
    print(io, "$nettype($layer_str)")
end

# DBN fields may change in the future, so it's worth to work through accessors
getname(net::Net, k::Int) = net.layernames[k]
getmodel(net::Net, k::Int) = net.layers[k]
function getmodel(net::Net, name::AbstractString)
    k = findfirst(lname -> lname == name, net.layernames)
    return (k != 0) ? net.layers[k] : error("No layer named '$name'")
end

# short syntax for accessing stored RBMs
getindex(net::Net, k::Int) = getmodel(net, k)
getindex(net::Net, name::AbstractString) = getmodel(net, name)
Base.length(net::Net) = length(net.layers)
Base.endof(net::Net) = length(net)


function hid_means_at_layer(net::Net, batch::Array{Float64, 2}, layer::Int)
    hiddens = Array{Array{Float64, 2}}(layer)
    hiddens[1] = hid_means(net[1], batch)
    for k=2:layer
        hiddens[k] = hid_means(net[k], hiddens[k-1])
    end
    hiddens[end]
end


function transform(net::Net, X::Mat{Float64})
    return hid_means_at_layer(net, X, length(net))
end


function fit(dbn::DBN, X::Mat{Float64}; ctx = Dict{Any,Any}())
    @assert minimum(X) >= 0 && maximum(X) <= 1
    n_samples = size(X,2)
    batch_size = @get(ctx, :batch_size, 100)
    n_batches = round(Int, ceil(n_samples / batch_size))
    n_epochs = @get(ctx, :n_epochs, 10)
    reporter = @get_or_create(ctx, :reporter, TextReporter())
    for k = 1:length(dbn.layers)
        for epoch=1:n_epochs
            report(reporter, dbn, epoch, k)
            for i=1:n_batches
                batch = X[:, ((i-1)*batch_size + 1):min(i*batch_size, end)]
                input = k == 1 ? batch : hid_means_at_layer(dbn, batch, k-1)
                fit_batch!(dbn[k], input, ctx)
            end
        end
    end
end

fit(dbn::DBN, X::Mat{T}; opts...) where {T} = fit(rbm, X, Dict(opts))


function invert(rbm::RBM)
    irbm = deepcopy(rbm)
    irbm.W = rbm.W'
    irbm.vbias = rbm.hbias
    irbm.hbias = rbm.vbias
    return irbm
end


function unroll(dbn::DBN)
    n = length(dbn)
    layers = Array{RBM}(2n)
    layernames = Array{String}(2n)
    layers[1:n] = dbn.layers
    layernames[1:n] = dbn.layernames
    for i=1:n
        layers[n+i] = invert(dbn[n-i+1])
        layernames[n+i] = getname(dbn, n-i+1) * "_inv"
    end
    return DAE(layers, layernames)
end

