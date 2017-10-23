import Boltzmann: split_evenly

if :unit in TEST_GROUPS
    @test split_evenly(10, 2) == [(1,2), (3,4), (5,6), (7,8), (9,10)]
    @test split_evenly(11, 2) == [(1,2), (3,4), (5,6), (7,8), (9,10), (11,11)]
    @test split_evenly(3, 5) == [(1,3)]
end
