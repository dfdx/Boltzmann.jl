if VERSION >= v"0.5-"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

using Boltzmann
using Benchmark
using DataFrames    # Needed for organization of benchmark results

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
