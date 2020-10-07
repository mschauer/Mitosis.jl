module Mitosis

using LinearAlgebra
using Random
using MeasureTheory

using MeasureTheory: AbstractMeasure

import Base: iterate, length
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


export Traced
struct Traced{T}
    x::T
end


include("gauss.jl")
include("markov.jl")
include("rules.jl")

end # module
