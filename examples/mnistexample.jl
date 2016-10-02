
using Boltzmann
using MNIST
using ImageView

function plot_weights(W, imsize, padding=10)
    h, w = imsize
    n = size(W, 1)
    rows = Int(floor(sqrt(n)))
    cols = Int(ceil(n / rows))
    halfpad = div(padding, 2)
    dat = zeros(rows * (h + padding), cols * (w + padding))
    for i=1:n
        wt = W[i, :]
        wim = reshape(wt, imsize)
        wim = wim ./ (maximum(wim) - minimum(wim))
        r = div(i - 1, cols) + 1
        c = rem(i - 1, cols) + 1
        dat[(r-1)*(h+padding)+halfpad+1 : r*(h+padding)-halfpad,
            (c-1)*(w+padding)+halfpad+1 : c*(w+padding)-halfpad] = wim
    end
    ImageView.view(dat)
    return dat
end


function run_mnist()
    X, y = testdata()  # test data is smaller, no need to downsample
    X = X ./ (maximum(X) - minimum(X))
    m = RBM(Degenerate, Bernoulli, 28*28, 300)
    fit(m, X, n_epochs=20, randomize=true)
    plot_weights(m.W[1:64, :], (28, 28))
    return m
end

run_mnist()

println("Press RETURN when ready")
readline(STDIN)

