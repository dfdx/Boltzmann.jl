using Base.Test
using Boltzmann
using BenchmarkTools

"""
Runs a series of basic smoke and benchmark tests on the interface of any RBM.

Default methods tested:

* `fit(model, X)`
* `transform(model, X)`
* `generate(model, X)`

Args:
* model::Union{AbstractRBM, Net} - the model to test

Optional:
* opts::Dict - the context passed when running fit. (default=DEFAULT_CONTEXT)
* n_obs::Int - total number of observations in generated dataset (default=1000)

Returns: BenchmarkGroup
"""
function benchmark!(rbm::AbstractRBM{T,V,H}, suite::BenchmarkGroup; opts::Dict=DefaultContext(), input_size=-1, debug=false, n_obs=(1000, 5000, 10000)) where {T,V,H}
    tname = String(typeof(rbm).name)
    suite[tname, String(T.name)] = BenchmarkGroup()

    if debug && isa(ctx[:reporter], TestReporter)
        ctx[:reporter].log = true
    end

    for m in n_obs
        ctx = deepcopy(opts)
        n_hid, n_vis = size(rbm.W)

        if input_size > 0
            n_vis = input_size
        end

        DX = generate_dataset(T, n_vis; n_obs=n_obs)
        SX = generate_dataset(T, n_vis; n_obs=n_obs, sparsity=0.3)


        for X in (DX, SX)
            suite[tname]["fit", size(X), String(typeof(X).name)] = @benchmarkable fit($rbm, $X, $ctx)
            suite[tname]["transform", size(X), String(typeof(X).name)] = @benchmarkable transform($rbm, $X)
            suite[tname]["generate", size(X), String(typeof(X).name)] = @benchmarkable generate($rbm, $X)
        end
    end

    return suite
end

#=
Available test groups are:
* :unit,
* :integration,
* :acceptance,
* :benchmark,
=#
const DEFAULT_TEST_GROUPS = Set([:unit, :integration])
TEST_GROUPS = DEFAULT_TEST_GROUPS

# If the JULIA_TEST_GROUPS ENV variable is set
# then use the inte
if haskey(ENV, "JULIA_TEST_GROUPS")
    TEST_GROUPS = Set(
        Symbol[split(ENV["JULIA_TEST_GROUPS"])...]
    )
end

tests = [
    "utils",
    "rbm",
    "nets",
    "conditional",
]

for t in tests
    println("\n\nRunning $t tests...\n")
    include(string("test", t, ".jl"))
end
