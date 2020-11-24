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



## References

* Frank van der Meulen, Moritz Schauer (2020): Automatic Backward Filtering Forward Guiding for Markov processes and graphical models. [[arXiv:2010.03509]](https://arxiv.org/abs/2010.03509).
