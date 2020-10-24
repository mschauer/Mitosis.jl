module Mitosis

using Statistics
using LinearAlgebra
using Random
using MeasureTheory

using MeasureTheory: AbstractMeasure, ScaledMeasure

import Base: iterate, length

export Gaussian
export Traced, BFFG, left′, right′, backwardfilter, forwardsampler



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

inner(x) = dot(x,x)
outer(x) = x*x'

_logdet(Σ, d::Integer) = LinearAlgebra.logdet(Σ)


macro F(f) :(::typeof($f)) end

function left′
end
function right′
end
backwardfilter(k, a) = right′(BFFG(), k, a)
forwardsampler(k, u, z, m) = left′(BFFG(), k, u, z, m)
sym(x) = Symmetric(x)


struct Traced{T}
    x::T
end


include("gauss.jl")
include("wgaussian.jl")
include("markov.jl")
include("rules.jl")

end # module
