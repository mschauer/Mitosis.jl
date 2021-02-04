module Mitosis

using UnPack
using Statistics, StatsBase
using LinearAlgebra
using Random
using MeasureTheory
import MeasureTheory: kernel

"""
    kernel(f, M)
    kernel((f1, f2, ...), M)

A kernel `κ = kernel(f, M)` returns a wrapper around
a function `f` giving the parameters for a measure of type `M`,
such that `κ(x) = M(f(x)...)`
respective `κ(x) = M(f1(x), f2(x), ...)`.

If the argument is a named tuple `(;a=f1, b=f1)`, `κ(x)` is defined as
`M(;a=f(x),b=g(x))`.

# Reference

* https://en.wikipedia.org/wiki/Markov_kernel
"""
kernel

using MeasureTheory: AbstractMeasure, WeightedMeasure
import MeasureTheory: density, logdensity
import Base: iterate, length
import Random.rand
import StatsBase.sample

export Gaussian, Copy, fuse, weighted
export Traced, BFFG, left′, right′, forward, backward, backwardfilter, forwardsampler
export BF, density, logdensity, ⊕, kernel, correct, Kernel, WGaussian, Gaussian, ConstantMap, AffineMap, LinearMap, GaussKernel


function independent_sum
end
const ⊕ = independent_sum

const ∅ = nothing

abstract type Context end

"""
    BFFG()

Backward filter forward guiding context for non-linear Gaussian
systems with `h` parametrized by `WGaussian{(:F,:Γ,:c)}`` (see Theorem 7.1 [Automatic BFFG].)
"""
struct BFFG <: Context
end

"""
    BF()

Backward filter for linear Gaussian systems parametrized
by mean and covariance of the backward filtered marginal distribution.
"""
struct BF <: Context
end

include("linearalgebra.jl")

macro F(f) :(::typeof($f)) end
struct Leaf{T}
    y::T
end
Base.getindex(y::Leaf) = y.y

struct Copy{N}
end
(a::Copy{2})(x) = (x, x)

function forward
end
function backward
end
const left′ = forward
const right′ = backward

backwardfilter(k, a; kargs...) = backward(BFFG(), k, a; kargs...)
forwardsampler(k, m, x; kargs...) = forward(BFFG(), k, m, x; kargs...)

fuse(a; kargs...) = backward(BFFG(), Copy{1}(), a; kargs...)
fuse(a, b; kargs...) = backward(BFFG(), Copy{2}(), a, b; kargs...)
fuse(a, b, c; kargs...) = backward(BFFG(), Copy{3}(), a, b, c; kargs...)

struct Traced{T}
    x::T
end

struct Weighted{S,T}
    x::S
    ll::T
end
weighted(x) = Weighted(x, 0.0)
weighted(x, ll) = Weighted(x, ll)
weighted(x::Weighted, ll) = Weighted(x.x, ll + x.ll)
Base.getindex(x::Weighted) = x.x
include("gauss.jl")
include("wgaussian.jl")
include("markov.jl")
include("rules.jl")
include("regression.jl")

function forward(bffg::BFFG, k, m, x::Weighted)
    p = forward(bffg, k, m)(x[])
    weighted(p, x.ll)
end
function forward(bffg::BFFG, k::Tuple, m, x::Weighted)
    p = forward(bffg, k..., m, x[])
    weighted(p, x.ll)
end


forwardsampler(k, k̃::Kernel, m, x; kargs...) = forward(BFFG(), k, k̃, m, x; kargs...)

end # module
