
using Boltzmann: split_evenly
using Base.Test

@assert split_evenly(10, 2) == [(1,2), (3,4), (5,6), (7,8), (9,10)]
@assert split_evenly(11, 2) == [(1,2), (3,4), (5,6), (7,8), (9,10), (11,11)]
@assert split_evenly(3, 5) == [(1,3)]
