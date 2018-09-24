
using HDF5


function save_params(file::HDF5File, rbm::RBM, name::AbstractString)
    write(file, "$(name)___weight", copy(rbm.W'))
    write(file, "$(name)___vbias", rbm.vbias)
    write(file, "$(name)___bias", rbm.hbias)
end

function load_params(file::HDF5File, rbm::RBM, name::AbstractString)
    rbm.W = read(file, "$(name)___weight")'
    rbm.vbias = read(file, "$(name)___vbias")
    rbm.hbias = read(file, "$(name)___bias")
end

function save_params(file::HDF5File, net::Net)
    for i=1:length(net)
        save_params(file, net[i], getname(net, i))
    end
end
save_params(path::AbstractString, net::Net) = h5open(path, "w") do h5
    save_params(h5, net)
end


function load_params(file, net::Net)
    for i=1:length(net)
        load_params(file, net[i], getname(net, i))
    end
end
load_params(path::AbstractString, net::Net) = h5open(path) do h5
    load_params(h5, net)
end


