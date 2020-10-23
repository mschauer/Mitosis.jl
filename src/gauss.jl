import Statistics: mean, cov
import Random.rand
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

const GaussianOrNdTuple{P} = Union{Gaussian{P},NamedTuple{P}}

Base.keys(p::Gaussian{P}) where {P} = P

## Basics

Base.:(==)(p1::Gaussian, p2::Gaussian) = mean(p1) == mean(p2) && cov(p1) == cov(p2)
Base.isapprox(p1::Gaussian, p2::Gaussian; kwargs...) =
    isapprox(mean(p1), mean(p2); kwargs...) && isapprox(cov(p1), cov(p2); kwargs...)

mean(p::Gaussian{(:μ, :Σ)}) = p.par.μ
cov(p::Gaussian{(:μ, :Σ)})  = p.par.Σ
meancov(p) = mean(p), cov(p)

precision(p::Gaussian{(:μ, :Σ)}) = inv(p.par.Σ)

mean(p::Gaussian{(:F, :Γ)}) = Γ\p.par.F
cov(p::Gaussian{(:F, :Γ)}) = inv(p.par.Γ)
precision(p::Gaussian{(:F, :Γ)}) = p.par.Γ

dim(p::Gaussian{(:F, :Γ)}) = length(p.par.F)
dim(p::Gaussian) = length(mean(p))
whiten(p::Gaussian{(:μ, :Σ)}, x) = cholesky(p.par.Σ).U'\(x - p.par.μ)
unwhiten(p::Gaussian{(:μ, :Σ)}, z) = cholesky(p.par.Σ).U'*z + p.par.μ
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
    Z = p.par.Σ[A,B]*inv(p.par.Σ[B,B])
    Gaussian{(:μ, :Σ)}(μ = p.par.μ[A] + Z*(xB - p.par.μ[B]), Σ = p.par.Σ[A,A] - Z*p.par.Σ[B,A])
end
