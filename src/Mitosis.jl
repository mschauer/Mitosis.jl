module Mitosis

using LinearAlgebra
using Random
using MeasureTheory

using MeasureTheory: AbstractMeasure, ScaledMeasure

import Base: iterate, length

export Traced, BFFG, left′, right′, backwardfilter, forwardsampler



function independent_sum
end
const ⊕ = independent_sum

const ∅ = nothing

abstract type Context end

struct BFFG <: Context
end

inner(x) = dot(x,x)
outer(x) = x*x'

logdet(Σ, d) = LinearAlgebra.logdet(Σ)


macro F(f) :(::typeof($f)) end

function left′
end
function right′
end
backwardfilter(k, a) = right′(BFFG(), k, a)
forwardsampler(k, u, z, m) = left′(BFFG(), k, u, z, m)



struct Traced{T}
    x::T
end


include("gauss.jl")
include("markov.jl")
include("rules.jl")

end # module
