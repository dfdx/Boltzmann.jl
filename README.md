
Restricted Bolzmann Machines in Julia
=====================================

This package provides implementation of 2 most commonly used types of Restricted Bolzmann Machines, namely: 

- **BernoulliRBM**: RBM with binary visible and hidden units
- **GRBM**: RBM with Gaussian visible and binary hidden units

Usage: 

    X = ...  # data matrix, observations as columns, variables as rows
    model = GRBM(n_visibles, n_hiddens)
    fit!(model, X, n_iter=10, n_gibbs=3, lr=0.1)
    model.weights[1:10, :]  # matrix of learned weights; unlike data matrix, learned components are on rows, not columns
