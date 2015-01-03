
using HDF5


function save_params(file::HDF5File, rbm::RBM, name::String)
    write(file, "$(name)___weight", rbm.W')
    write(file, "$(name)___vbias", rbm.vbias)
    write(file, "$(name)___bias", rbm.hbias)
end

function load_params(file::HDF5File, rbm::RBM, name::String)
    rbm.W = read(file, "$(name)___weight")'
    rbm.vbias = read(file, "$(name)___vbias")
    rbm.hbias = read(file, "$(name)___bias")
end

function save_params(file::HDF5File, dbn::DBN)
    for i=1:length(dbn)
        save_params(file, dbn[i], getname(dbn, i))
    end
end
save_params(path::String, dbn::DBN) = h5open(path, "w") do h5
    save_params(h5, dbn)
end


function load_params(file, dbn::DBN)
    for i=1:length(dbn)
        load_params(file, dbn[i], getname(dbn, i))
    end
end
load_params(path::String, dbn::DBN) = h5open(path) do h5
    load_params(h5, dbn)
end





function from_examples_test_jl()
    use_cuda = false
        
    srand(12345678)

    ############################################################
    # Prepare Random Data
    ############################################################
    N = 10000
    M = 20
    P = 10

    X = rand(M, N)
    W = rand(M, P)
    B = rand(P, 1)

    Y = (W'*X .+ B)
    Y = Y + 0.01*randn(size(Y))

    ############################################################
    # Define network
    ############################################################
    if use_cuda
        backend = GPUBackend()
    else
        backend = CPUBackend()
    end
    init(backend)
    
    data_layer = MemoryDataLayer(batch_size=500, data=Array[X,Y])
    weight_layer = InnerProductLayer(name="vis",output_dim=P, tops=[:pred], bottoms=[:data])
    loss_layer = SquareLossLayer(bottoms=[:pred, :label])

    net = Net("TEST", backend, [loss_layer, weight_layer, data_layer])
    println(net)

    ############################################################
    # Solve
    ############################################################
    lr_policy = LRPolicy.Staged(
                                  (6000, LRPolicy.Fixed(0.001)),
                                  (4000, LRPolicy.Fixed(0.0001)),
                                )
    params = SolverParameters(regu_coef=0.0005, mom_policy=MomPolicy.Fixed(0.9), max_iter=10000, lr_policy=lr_policy)
    solver = SGD(params)
    add_coffee_break(solver, TrainingSummary(), every_n_iter=100)

    solve(solver, net)

    learned_b = similar(B)
    copy!(learned_b, net.states[2].b)

    #println("$(learned_b)")
    #println("$(B)")

    shutdown(backend)
end



function main()
    N = 10000
    M = 20
    P = 10

    X = rand(M, N)
    W = rand(M, P)
    B = rand(P, 1)

    Y = (W'*X .+ B)
    Y = Y + 0.01*randn(size(Y))

    backend = CPUBackend()
    
    data_layer = MemoryDataLayer(batch_size=500, data=Array[X,Y])
    weight_layer = InnerProductLayer(name="ip",output_dim=P, tops=[:pred], bottoms=[:data])
    loss_layer = SquareLossLayer(bottoms=[:pred, :label])

    net = Net("TEST", backend, [loss_layer, weight_layer, data_layer])

    rbm = BernoulliRBM(20, 10)
    fit(rbm, X)

    save_as_mocha("model.hdf5", rbm)

    h5open("model.hdf5") do h5
        load_network(h5, net)
    end
    
end
