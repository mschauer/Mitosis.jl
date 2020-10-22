import Statistics: mean, cov
import Random.rand
struct Gaussian{P} <: AbstractMeasure
    par::P
end
Gaussian(;args...) = Gaussian(args.data)

Base.:(==)(p1::Gaussian, p2::Gaussian) = mean(p1) == mean(p2) && cov(p1) == cov(p2)
Base.isapprox(p1::Gaussian, p2::Gaussian; kwargs...) =
    isapprox(mean(p1), mean(p2); kwargs...) && isapprox(cov(p1), cov(p2); kwargs...)

mean(p::Gaussian{P}) where {P <: NamedTuple{(:μ, :Σ)}} = p.par.μ
cov(p::Gaussian{P}) where {P <: NamedTuple{(:μ, :Σ)}} = p.par.Σ
precision(p::Gaussian{P}) where {P <: NamedTuple{(:μ, :Σ)}} = inv(p.par.Σ)

mean(p::Gaussian{P}) where {P <: NamedTuple{(:F, :Γ)}} = Γ\p.par.F
cov(p::Gaussian{P}) where {P <: NamedTuple{(:F, :Γ)}} = inv(p.par.Γ)
precision(p::Gaussian{P}) where {P <: NamedTuple{(:F, :Γ)}} = p.par.Γ

dim(p::Gaussian{P}) where {P <: NamedTuple{(:F, :Γ)}} = length(p.par.F)
dim(p::Gaussian) = length(mean(p))
whiten(p::Gaussian{P}, x) where {P <: NamedTuple{(:μ, :Σ)}} = cholesky(p.par.Σ).U'\(x - p.par.μ)
unwhiten(p::Gaussian{P}, z) where {P <: NamedTuple{(:μ, :Σ)}} = cholesky(p.par.Σ).U'*z + p.par.μ
sqmahal(p::Gaussian, x) = norm_sqr(whiten(p, x))

rand(p::Gaussian) = rand(Random.GLOBAL_RNG, p)
rand(RNG::AbstractRNG, p::Gaussian) = unwhiten(p, randn!(RNG, zero(mean(p))))

MeasureTheory.logdensity(p::Gaussian, x) = -(sqmahal(p,x) + _logdet(p, dim(p)) + dim(p)*log(2pi))/2
MeasureTheory.density(p::Gaussian, x) = exp(logpdf(p, x))

## WeightedGaussian

weight(μ) = 0.0
weight(μ::ScaledMeasure) = logscale(μ)
weightedgaussian(c; args...) = MeasureTheory.ScaledMeasure(logscale, Gaussian(; args...))
wgaussian_params(p::Gaussian) = weight(p), cov(p), mean(p)

## Algebra

Base.:+(p::Gaussian{P}, x) where {P} = Gaussian{P}(mean(p) + x, p.par[2])
Base.:+(x, p::Gaussian) = p + x

Base.:-(p::Gaussian, x) = p + (-x)
Base.:*(M, p::Gaussian{P}) where {P <: NamedTuple{(:μ, :Σ)}}  = Gaussian{P}(M * mean(p), Σ = M * cov(p) * M')

⊕(p1::Gaussian{P}, p2::Gaussian{P}) where {P <: NamedTuple{(:μ, :Σ)}} = Gaussian{P}(p1.μ + p2.μ, p1.Σ + p2.Σ)
⊕(x, p::Gaussian) = x + p
⊕(p::Gaussian, x) = p + x

## Conditional

"""
    conditional(P::Gaussian, A, B, xB)
Conditional distribution of `X[i for i in A]` given
`X[i for i in B] == xB` if ``X ~ P``.
"""
function conditional(P::Gaussian{T}, A, B, xB) where {T <: NamedTuple{(:μ, :Σ)}}
    Z = P.par.Σ[A,B]*inv(P.par.Σ[B,B])
    Gaussian(μ = P.par.μ[A] + Z*(xB - P.par.μ[B]), Σ = P.par.Σ[A,A] - Z*P.par.Σ[B,A])
end
