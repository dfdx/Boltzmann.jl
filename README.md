
Boltzmann.jl
============

[![Build Status](https://travis-ci.org/dfdx/Boltzmann.jl.svg)](https://travis-ci.org/dfdx/Boltzmann.jl)

Restricted Boltzmann machines and deep belief networks in Julia

Installation
------------

    Pkg.add("Boltzmann")

installing latest development version: 

    Pkg.clone("https://github.com/dfdx/Boltzmann.jl")


RBM Basic Usage
---------------

Traint RBM:

    using Boltzmann

    X = randn(100, 2000)    # 2000 observations (examples) 
                            #  with 100 variables (features) each
    X = (X + abs(minimum(X))) / (maximum(X) - minimum(X)) # scale X to [0..1]
    rbm = GRBM(100, 50)     # define Gaussian RBM with 100 visible (input) 
                            #  and 50 hidden (output) variables
    fit(rbm, X)             # fit model to data 

(for more meaningful dataset see [MNIST Example](https://github.com/dfdx/Boltzmann.jl/blob/master/examples/mnistexample.jl))

After model is fitted, you can **extract learned components (a.k.a. weights)**: 

    comps = components(rbm)
    
**transform** data vectors into new higher-level representation (e.g. for further classification): 

    Xt = transform(rbm, X)  # vectors of X have length 100, vectors of Xt - length 50

or **generate** vectors similar to given ones (e.g. for recommendation, see example [here](https://github.com/dfdx/lastfm-rbm))

    x = ... 
    x_new = generate(rbm, x)

RBMs can handle both - dense and sparse arrays. It cannot, however, handle DataArrays because it's up to application how to treat missing value.


RBM Kinds
---------

This package provides implementation of the 2 most popular kinds of restricted Boltzmann machines: 

 - `BernoulliRBM`: RBM with binary visible and hidden units
 - `GRBM`: RBM with Gaussian visible and binary hidden units

Bernoulli RBM is classic one and works great for modeling binary (e.g. like/dislike) and nearly binary (e.g. logsitic-based) data. Gaussian RBM works better when visible variables approximately follow normal distribution, which is often the case e.g. for image data. 


Deep Belief Networks
--------------------

DBNs are created as a stack of named RBMs. Below is an example of training DBN for MNIST dataset:

    using Boltzmann
    using MNIST

    X, y = traindata()
    X = X[:, 1:1000]                     # take only 1000 observations for speed
    X = X / (maximum(X) - (minimum(X)))  # normalize to [0..1]

    layers = [("vis", GRBM(784, 256)),
              ("hid1", BernoulliRBM(256, 100)),
              ("hid2", BernoulliRBM(100, 100))]
    dbn = DBN(layers)
    fit(dbn, X)
    transform(dbn, X)


Integration with Mocha
----------------------

[Mocha.jl](https://github.com/pluskid/Mocha.jl) is an excellent deep learning framework implementing auto-encoders and a number of fine-tuing algorithms. Boltzmann.jl allows to save pretrained model in a Mocha-compatible file format to be used later on for supervised learning. Below is a snippet of essential API, while complete code is available in [Mocha Export Example](https://github.com/dfdx/Boltzmann.jl/blob/master/examples/mocha_export_example.jl):

    # pretraining and exporting in Boltzmann.jl
    dbn_layers = [("vis", GRBM(100, 50)),
                  ("hid1", BernoulliRBM(50, 25)),
                  ("hid2", BernoulliRBM(25, 20))]
    dbn = DBN(dbn_layers)
    fit(dbn, X)
    save_params(DBN_PATH, dbn)

    # loading in Mocha.jl
    backend = CPUBackend()
    data = MemoryDataLayer(tops=[:data, :label], batch_size=500, data=Array[X, y])
    vis = InnerProductLayer(name="vis", output_dim=50, tops=[:vis], bottoms=[:data])
    hid1 = InnerProductLayer(name="hid1", output_dim=25, tops=[:hid1], bottoms=[:vis])
    hid2 = InnerProductLayer(name="hid2", output_dim=20, tops=[:hid2], bottoms=[:hid1])
    loss = SoftmaxLossLayer(name="loss",bottoms=[:hid2, :label])
    net = Net("TEST", backend, [data, vis, hid1, hid2])

    h5open(DBN_PATH) do h5
        load_network(h5, net)
    end



