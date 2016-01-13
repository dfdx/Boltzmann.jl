
include("utils.jl")

abstract AbstractRBM

@runonce type RBM{V,H} <: AbstractRBM
    W::Matrix{Float64}
    vbias::Vector{Float64}
    hbias::Vector{Float64}
end

function RBM(V::Type, H::Type,
             n_vis::Int, n_hid::Int; sigma=0.01)
    RBM{V,H}(rand(Normal(0, sigma), (n_hid, n_vis)),
        zeros(n_vis), zeros(n_hid))
end

function Base.show{V,H}(io::IO, rbm::RBM{V,H})
    n_vis = size(rbm.vbias, 1)
    n_hid = size(rbm.hbias, 1)
    print(io, "RBM{$V,$H}($n_vis, $n_hid)")
end


function persistent_contdiv()
end

function default_update!()
end


function fit(rbm::RBM, X::Matrix{Float64};
             gradient=persistent_contdiv,update! = default_update!, opts...)
    println(Dict{Symbol,Any}(opts))
end


