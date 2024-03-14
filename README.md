[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mschauer.github.io/Mitosis.jl)
[![Build Status](https://github.com/mschauer/Mitosis.jl/workflows/CI/badge.svg)](https://github.com/mschauer/Mitosis.jl/actions)

# Mitosis.jl (work in progress)

Incorporate discrete and continuous time Markov processes as building blocks into probabilistic graphical models.

## Synopsis

Mitosis implements the backward filter and the forward change of measure  of the Automatic Backward Filtering Forward Guiding paradigm  (van der Meulen and Schauer, 2020) as transformation rules for general generative models,  suitable to be incorporated into probabilistic programming approaches.

Starting point is the generative model, a forward description of the probabilistic process dynamics. The backward filter backpropagate the information provided by observations through the model to transform the generative (forward) model into a preconditional model guided by the data.

The preconditional model approximates the actual conditional model, with known likelihood-ratio between the two and can be used in a variety of sampling and statistical inference approaches to speed up inference.

## Overview

This package will contain the general infrastructure and the rules for non-linear Gaussian transition kernels, thus allowing non-linear state space models.

In parallel, rules for stochastic differential equations are developed in [MitosisStochasticDiffEq.jl](https://github.com/mschauer/MitosisStochasticDiffEq.jl)

## Show reel

### Bayesian regression on the drift parameter of an SDE
```julia
using StochasticDiffEq
using Random
using MitosisStochasticDiffEq
import MitosisStochasticDiffEq as MSDE
using LinearAlgebra, Statistics

# Model and sensitivity
function f(du, u, θ, t)
    c = 0.2 * θ
    du[1] = -0.1 * u[1] + c * u[2]
    du[2] = - c * u[1] - 0.1 * u[2]
    return
end
function g(du, u, θ, t)
    fill!(du, 0.15)
    return
end

# b is linear in the parameter with Jacobian 
function b_jac(J,x,θ,t)
    J .= false
    J[1,1] =   0.2 * x[2] 
    J[2,1] = - 0.2 * x[1]
    nothing
end
# and intercept
function b_icpt(dx,x,θ,t)
    dx .= false
    dx[1] = -0.1 * x[1]
    dx[2] = -0.1 * x[2]
    nothing
end

# Simulate path ensemble 
x0 = [1.0, 1.0]
tspan = (0.0, 20.0)
θ0 = 1.0
dt = 0.05
t = range(tspan...; step=dt)

prob = SDEProblem{true}(f, g, x0, tspan, θ0)
ensembleprob = EnsembleProblem(prob)
ensemblesol = solve(
    ensembleprob, EM(), EnsembleThreads(); dt=dt, saveat=t, trajectories=1000
)

# Inference on drift parameters
sdekernel = MSDE.SDEKernel(f,g,t,0*θ0)
ϕprototype = zeros((length(x0),length(θ0))) # prototypes for vectors
yprototype = zeros((length(x0),))
R = MSDE.Regression!(sdekernel,yprototype,paramjac_prototype=ϕprototype,paramjac=b_jac,intercept=b_icpt)
prior_precision = 0.1I(1)
posterior = MSDE.conjugate(R, ensemblesol, prior_precision)
print(mean(posterior)[], " ± ", sqrt(cov(posterior)[]))
```


## References

* Frank van der Meulen, Moritz Schauer (2020): Automatic Backward Filtering Forward Guiding for Markov processes and graphical models. [[arXiv:2010.03509]](https://arxiv.org/abs/2010.03509).
