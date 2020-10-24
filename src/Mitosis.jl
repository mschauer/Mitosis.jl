module Mitosis

using UnPack
using Statistics
using LinearAlgebra
using Random
using MeasureTheory

using MeasureTheory: AbstractMeasure, ScaledMeasure

import Base: iterate, length

export Gaussian, Copy
export Traced, BFFG, left′, right′, backwardfilter, forwardsampler
export BF, logdensity, ⊕, kernel, correct, Kernel, WGaussian, Gaussian, ConstantMap, AffineMap, LinearMap, GaussKernel


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
backwardfilter(k, a) = right′(BFFG(), k, a)
forwardsampler(k, u, z, m) = left′(BFFG(), k, u, z, m)
fuse(args::NTuple{N}...) where {N} = right′(BFFG(), Copy{N}(), args...)

struct Traced{T}
    x::T
end


include("gauss.jl")
include("wgaussian.jl")
include("markov.jl")
include("rules.jl")

end # module
