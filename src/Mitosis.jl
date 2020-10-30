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
backwardfilter(k, a; args...) = right′(BFFG(), k, a; args...)
forwardsampler(k, u, z, m; args...) = left′(BFFG(), k, u, z, m; args...)
fuse(a) = right′(BFFG(), Copy{1}(), a)
fuse(a,b) = right′(BFFG(), Copy{2}(), a, b)
fuse(a,b,c) = right′(BFFG(), Copy{3}(), a, b, c)

struct Traced{T}
    x::T
end


include("gauss.jl")
include("wgaussian.jl")
include("markov.jl")
include("rules.jl")

end # module
