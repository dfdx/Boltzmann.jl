import Boltzmann: split_evenly

if :unit in TEST_GROUPS
    @test split_evenly(10, 2) == [(1,2), (3,4), (5,6), (7,8), (9,10)]
    @test split_evenly(11, 2) == [(1,2), (3,4), (5,6), (7,8), (9,10), (11,11)]
    @test split_evenly(3, 5) == [(1,3)]
end

if :integration in TEST_GROUPS
    # Ensure that the test methods all work correctly
    df1 = Boltzmann.compare(GRBM(10, 5); n_obs=10)
    @test size(df1, 1) > 0

    # Try a comparison with 2 different contexts
    # in this case just use 2 different reporters
    context1 = Boltzmann.DefaultContext()
    context1[:reporter].log = true
    context2 = Boltzmann.DefaultContext()
    context2[:reporter] = Boltzmann.TextReporter()
    df2 = Boltzmann.compare(GRBM(10, 5), context1, context2; input_size=10, n_obs=10)
    @test size(df2, 1) > 0

    # Test the benchmark function (since we won't be running
    # it under coverage normally.
    df3 = Boltzmann.benchmark(GRBM(10, 5), input_size=10, n_obs=10, debug=true)
    @test size(df3, 1) > 0
end
