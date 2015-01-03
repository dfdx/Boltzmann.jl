
using Boltzmann
using Mocha
using HDF5

const DBN_PATH = "/tmp/dbn_params.hdf5"

# prepare training data
X = rand(100, 2000)
y = rand(2000)

# construct and train DBN
dbn_layers = [("vis", GRBM(100, 50)),
              ("hid1", BernoulliRBM(50, 25)),
              ("hid2", BernoulliRBM(25, 20))]
dbn = DBN(dbn_layers)
fit(dbn, X)

# save learn parameters (weights and biases)
if isfile(DBN_PATH)
    rm(DBN_PATH)
end
save_params(DBN_PATH, dbn)

# prepare Mocha network with similar architecture, but with additional layers
# for data and softmax classification
# names of layers should be the same as in DBN
backend = CPUBackend()
data = MemoryDataLayer(tops=[:data, :label], batch_size=500, data=Array[X, y])
vis = InnerProductLayer(name="vis", output_dim=50, tops=[:vis], bottoms=[:data])
hid1 = InnerProductLayer(name="hid1", output_dim=25, tops=[:hid1], bottoms=[:vis])
hid2 = InnerProductLayer(name="hid2", output_dim=20, tops=[:hid2], bottoms=[:hid1])
loss = SoftmaxLossLayer(name="loss",bottoms=[:hid2, :label])

net = Net("TEST", backend, [data, vis, hid1, hid2])

# finally, load pretrained parameters into corresponding layers of Mocha network
h5open(DBN_PATH) do h5
    load_network(h5, net)
    # if architecture of Mocha network doesn't match DBN exactly, one can
    # additonally pass `die_if_not_found=false` to ignore missing layers and
    # use default initializers instead
    #  load_network(h5, net, die_if_not_found=false) 
end

rm(DBN_PATH)
