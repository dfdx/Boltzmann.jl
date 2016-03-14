#=
The following file contains various function to help smoke test and benchmark
arbitrary AbstractRBMs and configurations.
=#

using Base.Test

"""
The TestReporter uses the ratio test or D'Alembert's criterion
to monitor convergence. This is helpful in automating tests that
confirm that a given RBM is learning something.

NOTE: current we just store the calculated ratios from each epoch,
but we could probably just do an online mean calculation.
"""
type TestReporter
    ratios::Vec{Float64}
    prev::Float64
    log::Bool

    function TestReporter(log=false)
        new(Array{Float64}(0), NaN, log)
    end
end

"""
Computes and adds a new ratio to the reporter. If log is set to
true in the reporter a message containing the score, time and mean ratio
will be printed.
"""
function report(r::TestReporter, rbm::AbstractRBM,
                epoch::Int, epoch_time::Float64, score::Float64)
    if epoch == 1
        r.ratios = Array{Float64}(0)
        r.prev = NaN
    end

    if !isnan(r.prev)
        # We're ratio test, also known as D'Alembert's criterion
        ratio = r.prev == 0.0 ? abs(score / nextfloat(r.prev)) : abs(score / r.prev)

        push!(r.ratios, ratio)
    end
    r.prev = score

    if r.log
        println("[Epoch $epoch] Score: $score [$(epoch_time)s] (Mean Ratio: $(mean(r.ratios)) )")
    end
end

"""
Simply return true or false as to whether the mean ratio is less than 1
"""
converged(r::TestReporter) = mean(r.ratios) < 1.0

"""
Simply returns a DefaultContext Dict.
"""
function DefaultContext()
    return Dict(
        :weight_decay_kind => :l2,
        :weight_decay_rate => 0.001,
        :sparsity_cost => 0.01,
        :sparsity_target => 0.02,
        :lr => 0.1,
        :momentum => 0.9,
        :batch_size => 100,
        :n_epochs => 10,
        :n_gibbs => 5,
        :reporter => TestReporter()
    )
end

"""
Generates synthetic random datasets with several modifiable properties.

Args:
* T::Type - the type (or precision) of the resulting matrix
* n_features::Int - the number of features in each observation

Optional:
* n_classes::Int - the number of unique classes or categories in the resulting dataset. (default=10)
* n_obs::Int - total number of observations in resulting dataset. (default=1000)
* sparsity::Float64 - specifies the density if a sparse matrix is desired. (default=-1.0)
                      if less than 0.0 a dense matrix is created.
* binary::Bool - whether or not to round the result to 0.0 or 1.0

Returns:
* Mat{T}(n_features, n_obs) - with the various properties specified.
"""
function generate_dataset(T::Type, n_features; n_classes=10, n_obs=1000, sparsity=-1.0, binary=true)
    function rand_fill!(X::Mat, prototypes::Mat)
        for i in 1:size(X, 2)
            j = rand(1:size(prototypes, 2))
            X[:, i] = prototypes[:, j]
        end

        if binary
            return round(X)
        else
            return X
        end
    end

    if sparsity > 0
        prototypes = map(T, sprand(n_features, n_classes, sparsity))
        X = map(T, spzeros(n_features, n_obs))
        return rand_fill!(X, prototypes)
    else
        prototypes = rand(T, n_features, n_classes)
        X = zeros(n_features, n_obs)
        return rand_fill!(X, prototypes)
    end
end

"""
Runs simple smoke tests on the provided RBM.

Args:
* rbm::AbstractRBM - the rbm to test

Optional:
* opts::Dict - an alternate context to use when calling `fit`. (default=DEFAULT_CONTEXT)
* n_obs::Int - the number of observations to generate for the synthetic datasets. (default=1000)
* debug::Bool - whether or not to print each epoch. (default=false)
                Only applies if the reporter in `ctx` is `TestReporter`

NOTE: Only using dense arrays for the dataset cause the conditional rbm doesn't support
sparse ones yet.
"""
function test{T,V,H}(rbm::AbstractRBM{T,V,H}; opts::Dict=DefaultContext(), input_size=-1, n_obs=1000, debug=false)
    n_hid, n_vis = size(rbm.W)

    if input_size > 0
        n_vis = input_size
    end

    ctx = deepcopy(opts)
    DX = generate_dataset(T, n_vis; n_obs=n_obs)
    SX = generate_dataset(T, n_vis; n_obs=n_obs, sparsity=0.3)

    if debug && isa(ctx[:reporter], TestReporter)
        ctx[:reporter].log = true
    end

    for X in (DX, SX)
        info("Testing against $(typeof(X).name)")
        info("Testing fit")
        fit(rbm, X, ctx)

        info("Testing for convergence")
        @test converged(ctx[:reporter])

        # Check that none of the array fields contain Infs or NaNs
        for name in fieldnames(rbm)
            if isa(fieldtype(typeof(rbm), name), AbstractArray)
                field = getfield(rbm, name)

                @test !any(isnan, field) && !any(isinf, field)
            end
        end

        info("Testing transform")
        Y = transform(rbm, X)
        @test !any(isnan, Y) && !any(isinf, Y)

        info("Testing generate")
        Y = generate(rbm, X)
        @test !any(isnan, Y) && !any(isinf, Y)
    end
end

"""
Runs a benchmark comparison between various rbms on a given function.

Args:
* func::Function - the function to compare
* rbms::Array{R<:AbstractRBM,1} - the RBMs to test
* args::Array{T<:Tuple,1} - the arguments to pass to each RBM's function

Returns: a DataFrame of the results
"""
function compare{R<:AbstractRBM, T<:Tuple}(func::Function, rbms::Array{R,1}, args::Array{T,1})
    @eval import Benchmark
    @assert length(rbms) == length(args)

    funcs = Array{Function}(length(rbms))
    names = Array{TypeName}(length(rbms))

    for i in eachindex(rbms)
        funcs[i] = () -> func(rbms[i], args[i]...)
        names[i] = typeof(rbms[i]).name
    end

    df = Benchmark.compare(funcs, 1)
    df[:Function] = fill(last(split(string(func), ".")), length(funcs))
    df[:RBM] = names

    return df
end

"""
Runs a full benchmark comparison on the public interface of an RBM
against those in Boltzmann.jl

Args:
* rbm::AbstractRBM - the rbm to test

Optional:
* n_obs::Int - number of observation in the synthetic dataset. (default=1000)

Returns: a DataFrame of the results

NOTE: For now we're only using dense matrices cause the conditional rbm doesn't
work with sparse ones.
"""
function compare{T,V,H}(rbm::AbstractRBM{T,V,H}; opts::Dict=DefaultContext(), input_size=-1, debug=false, n_obs=1000)
    @eval using DataFrames

    ctx = deepcopy(opts)
    models = AbstractRBM[rbm]
    n_hid, n_vis = size(rbm.W)

    if input_size > 0
        n_vis = input_size
    end

    DX = generate_dataset(T, n_vis; n_obs=n_obs)
    SX = generate_dataset(T, n_vis; n_obs=n_obs, sparsity=0.3)

    if !isa(rbm, RBM)
        push!(models, RBM(T, V, H, n_vis, n_hid))
    end

    if !isa(rbm, ConditionalRBM)
        n_cond = round(Int, n_vis / 2)
        push!(models, ConditionalRBM(T, V, H, n_vis - n_cond, n_hid, n_cond))
    end

    n = length(models)
    results = DataFrame()

    if debug && isa(ctx[:reporter], TestReporter)
        ctx[:reporter].log = true
    end

    for X in (DX, SX)
        fit_result = compare(fit, models, fill((X, deepcopy(ctx)), n))
        trans_result = compare(transform, models, fill((X,), n))

        tmp_result = vcat(
            fit_result,
            trans_result
        )

        tmp_result[:Input] = fill(typeof(X).name, n * 2)
        tmp_result[:Type] = fill(T, n * 2)
        results = vcat(results, tmp_result)
    end

    return results
end

"""
Runs a benchmark comparison on an RBM's fit method with different contexts.
This can be useful if you want benchmark various update or gradient methods
specified in the context.

Args:
* rbm::AbstractRBM - the rbm to test
* options::Dict... - the contexts to compare

Optional:
* n_obs::Int - the number of observations to generate in the synthetic dataset.

Returns: a DataFrame of the results
"""
function compare{T,V,H}(rbm::AbstractRBM{T,V,H}, options::Dict...; input_size=-1, n_obs=1000)
    n_hid, n_vis = size(rbm.W)
    if input_size > 0
        n_vis = input_size
    end

    X = generate_dataset(T, n_vis; n_obs=n_obs)
    args = [map(ctx -> (X, opt), options)]
    return compare(fit, [rbm, deepcopy(rbm)], args)
end


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

Returns: a DataFrame of the results
"""
function benchmark{T,V,H}(rbm::AbstractRBM{T,V,H}; opts::Dict=DefaultContext(), input_size=-1, debug=false, n_obs=1000)
    @eval begin
        using DataFrames
        import Benchmark
    end

    ctx = deepcopy(opts)
    n_hid, n_vis = size(rbm.W)

    if input_size > 0
        n_vis = input_size
    end

    DX = generate_dataset(T, n_vis; n_obs=n_obs)
    SX = generate_dataset(T, n_vis; n_obs=n_obs, sparsity=0.3)

    if debug && isa(ctx[:reporter], TestReporter)
        ctx[:reporter].log = true
    end

    results = DataFrame()

    for X in (DX, SX)
        # For now we're just benchmarking the fit method
        # but we could also get transform, generate, etc
        fit_func() = fit(rbm, X, ctx)
        trans_func() = transform(rbm, X)
        gen_func() = generate(rbm, X)

        results = vcat(results, Benchmark.benchmark(fit_func, "fit", "$(typeof(rbm).name):$(typeof(X).name)", 10))
        results = vcat(results, Benchmark.benchmark(trans_func, "transform", "$(typeof(rbm).name):$(typeof(X).name)", 10))
        results = vcat(results, Benchmark.benchmark(gen_func, "generate", "$(typeof(rbm).name):$(typeof(X).name)", 10))
    end

    return results
end
