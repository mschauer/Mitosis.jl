import Statistics: mean, cov
import Random.rand
import LinearAlgebra.logdet

"""
    Gaussian{(:μ,:Σ)}
    Gaussian{(:F,:Γ)}

Mitosis provides the measure `Gaussian` based on MeasureTheory.jl,
with a mean `μ` and covariance `Σ` parametrization,
or parametrised by natural parameters `F = Γ μ`, `Γ = Σ⁻¹`.

# Usage:

    Gaussian(μ=m, Σ=C)
    p = Gaussian{(:μ,:Σ)}(m, C)
    Gaussian(F=C\\m, Γ=inv(C))

    convert(Gaussian{(:F,:Γ)}, p)

    rand(rng, p)
"""
struct Gaussian{P,T} <: AbstractMeasure
    par::NamedTuple{P,T}
end
Gaussian{P}(nt::NamedTuple{P,T}) where {P,T} = Gaussian{P,T}(nt)
Gaussian{P}(args...) where {P} = Gaussian(NamedTuple{P}(args))
Gaussian(;args...) = Gaussian(args.data)
Gaussian{P}(;args...) where {P} = Gaussian{P}(args.data)

# the following propagates uncertainty if `μ` is `Gaussian`
Gaussian(par::NamedTuple{(:μ,:Σ),Tuple{T,S}}) where {T<:Gaussian,S} = Gaussian((;μ = mean(par.μ),Σ=par.Σ +cov(par.μ)))
Gaussian{P}(par::NamedTuple{(:μ,:Σ),Tuple{T,S}}) where {P,T<:Gaussian,S} = Gaussian{P}((;μ = mean(par.μ),Σ=par.Σ +cov(par.μ)))

Base.getproperty(p::Gaussian, s::Symbol) = getproperty(getfield(p, :par), s)

const GaussianOrNdTuple{P} = Union{Gaussian{P},NamedTuple{P}}

Base.keys(p::Gaussian{P}) where {P} = P
params(p::Gaussian) = getfield(p, :par)
## Basics

Base.:(==)(p1::Gaussian, p2::Gaussian) = mean(p1) == mean(p2) && cov(p1) == cov(p2)
Base.isapprox(p1::Gaussian, p2::Gaussian; kwargs...) =
    isapprox(mean(p1), mean(p2); kwargs...) && isapprox(cov(p1), cov(p2); kwargs...)

mean(p::Gaussian{(:μ, :Σ)}) = p.μ
mean(p::Gaussian{(:Σ,)}) = Zero()
cov(p::Gaussian{(:μ, :Σ)})  = p.Σ
meancov(p) = mean(p), cov(p)

precision(p::Gaussian{(:μ, :Σ)}) = inv(p.Σ)

mean(p::Gaussian{(:F, :Γ)}) = p.Γ\p.F
cov(p::Gaussian{(:F, :Γ)}) = inv(p.Γ)
precision(p::Gaussian{(:F, :Γ)}) = p.Γ
norm_sqr(x) = dot(x,x)
dim(p::Gaussian{(:F, :Γ)}) = length(p.F)
dim(p::Gaussian) = length(mean(p))
dim(p::Gaussian{(:Σ,)}) = size(p.Σ, 1)
whiten(p::Gaussian{(:μ, :Σ)}, x) = lchol(p.Σ)\(x - p.μ)
unwhiten(p::Gaussian{(:μ, :Σ)}, z) = lchol(p.Σ)*z + p.μ
whiten(p::Gaussian{(:Σ,)}, x) = lchol(p.Σ)\x
unwhiten(p::Gaussian{(:Σ,)}, z) = lchol(p.Σ)*z
sqmahal(p::Gaussian, x) = norm_sqr(whiten(p, x))

rand(p::Gaussian) = rand(Random.GLOBAL_RNG, p)
randwn(rng::AbstractRNG, x::Vector) = randn!(rng, zero(x))
randwn(rng::AbstractRNG, x) = map(xi -> randn(rng, typeof(xi)), x)

rand(rng::AbstractRNG, p::Gaussian) = unwhiten(p, randwn(rng, mean(p)))

_logdet(p::Gaussian{(:μ,:Σ)}) = _logdet(p.Σ, dim(p))
_logdet(p::Gaussian{(:Σ,)}) = logdet(p.Σ)
MeasureTheory.logdensity(p::Gaussian, x) = -(sqmahal(p,x) + _logdet(p) + dim(p)*log(2pi))/2
MeasureTheory.density(p::Gaussian, x) = exp(logdensity(p, x))
function MeasureTheory.logdensity(p::Gaussian{(:F,:Γ)}, x)
    C = cholesky(sym(p.Γ))
    -x'*p.Γ*x/2 + x'*p.F - p.F'*(C\p.F)/2  + logdet(C)/2 - dim(p)*log(2pi)/2
end

function Base.convert(::Type{Gaussian{(:F,:Γ)}}, p::Gaussian{(:μ,:Σ)})
    Γ = inv(p.Σ)
    return Gaussian{(:F,:Γ)}(Γ*p.μ, Γ)
end
function Base.convert(::Type{Gaussian{(:μ,:Σ)}}, p::Gaussian{(:F,:Γ)})
    Σ = inv(p.Γ)
    return Gaussian{(:μ,:Σ)}(Σ*p.F, Σ)
end

## Algebra

Base.:+(p::Gaussian{P}, x) where {P} = Gaussian{P}(mean(p) + x, p.par[2])
Base.:+(x, p::Gaussian) = p + x

Base.:-(p::Gaussian, x) = p + (-x)
Base.:*(M, p::Gaussian{P}) where {P} = Gaussian{P}(M * mean(p), Σ = M * cov(p) * M')

⊕(p1::Gaussian{(:μ, :Σ)}, p2::Gaussian{(:μ, :Σ)}) = Gaussian{(:μ, :Σ)}(p1.μ + p2.μ, p1.Σ + p2.Σ)
⊕(x, p::Gaussian) = x + p
⊕(p::Gaussian, x) = p + x

## Conditionals and filtering

"""
    conditional(p::Gaussian, A, B, xB)

Conditional distribution of `X[i for i in A]` given
`X[i for i in B] == xB` if ``X ~ P``.
"""
function conditional(p::Gaussian{(:μ, :Σ)}, A, B, xB)
    Z = p.Σ[A,B]*inv(p.Σ[B,B])
    Gaussian{(:μ, :Σ)}(p.μ[A] + Z*(xB - p.μ[B]), p.Σ[A,A] - Z*p.Σ[B,A])
end
