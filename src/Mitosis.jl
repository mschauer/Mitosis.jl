module Mitosis

using UnPack
using Statistics, StatsBase
using LinearAlgebra
using Random
using MeasureTheory

using MeasureTheory: AbstractMeasure, ScaledMeasure
import MeasureTheory: density, logdensity
import Base: iterate, length
import Random.rand
import StatsBase.sample

export Gaussian, Copy, fuse
export Traced, BFFG, left′, right′, backwardfilter, forwardsampler
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
struct Copy{N}
end
(a::Copy{2})(x) = (x, x)

function left′
end
function right′
end
backwardfilter(k, a; kargs...) = right′(BFFG(), k, a; kargs...)
forwardsampler(k, m, x; kargs...) = left′(BFFG(), k, m, x; kargs...)

fuse(a; kargs...) = right′(BFFG(), Copy{1}(), a; kargs...)
fuse(a, b; kargs...) = right′(BFFG(), Copy{2}(), a, b; kargs...)
fuse(a, b, c; kargs...) = right′(BFFG(), Copy{3}(), a, b, c; kargs...)

struct Traced{T}
    x::T
end


include("gauss.jl")
include("wgaussian.jl")
include("markov.jl")
include("rules.jl")

forwardsampler(k, k̃::Kernel, m, x; kargs...) = left′(BFFG(), k, k̃, m, x; kargs...)

end # module
