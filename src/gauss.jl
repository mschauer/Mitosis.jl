import Statistic: mean, cov
import Random.rand
struct Gaussian{P} <: AbstractMeasure
    par::P
end


Base.:(==)(g1::Gaussian, g2::Gaussian) = mean(g1) == mean(g2) && cov(g1) == cov(g2)
Base.isapprox(g1::Gaussian, g2::Gaussian; kwargs...) =
    isapprox(mean(g1), mean(g2); kwargs...) && isapprox(cov(g1), cov(g2); kwargs...)

mean(P::Gaussian{P}) where {P <: NamedTuple{(:μ, :Σ)}} = mean(P)
cov(P::Gaussian{P}) where {P <: NamedTuple{(:μ, :Σ)}} = P.Σ
precision(P::Gaussian{P}) where {P <: NamedTuple{(:μ, :Σ)}} = inv(P.Σ)

mean(P::Gaussian{P}) where {P <: NamedTuple{(:F, :Γ)}} = Γ\P.F
cov(P::Gaussian{P}) where {P <: NamedTuple{(:F, :Γ)}} = inv(P.Γ)
precision(P::Gaussian{P}) where {P <: NamedTuple{(:F, :Γ)}} = P.Γ

dim(P::Gaussian{P}) where {P <: NamedTuple{(:F, :Γ)}} = length(P.F)
dim(P::Gaussian) = length(mean(P))
whiten(P::Gaussian{P}, x) where {P <: NamedTuple{(:μ, :Σ)}} = cholesky(P.Σ).U'\(x - P.μ)
unwhiten(P::Gaussian{P}, z) where {P <: NamedTuple{(:μ, :Σ)}} = cholesky(P.Σ).U'*z + P.μ
sqmahal(P::Gaussian, x) = norm_sqr(whiten(P, x))

rand(P::Gaussian) = rand(GLOBAL_RNG, P)
rand(RNG::AbstractRNG, P::Gaussian) = unwhiten(P, randn(RNG, typeof(mean(P))))

MeasureTheory.logdensity(P::Gaussian{P}, x) = -(sqmahal(P,x) + _logdet(P, dim(P)) + dim(P)*log(2pi))/2
MeasureTheory.density(P::Gaussian, x) = exp(logpdf(P, x))

Base.:+(g::Gaussian{P}, x) = Gaussian{P}(mean(g) + x, g.par[2])
Base.:+(x, g::Gaussian) = g + x

Base.:-(g::Gaussian, x) = g + (-x)
Base.:*(M, g::Gaussian{P}) where {P <: NamedTuple{(:μ, :Σ)}}  = Gaussian{P}(M * mean(g), Σ = M * cov(g) * M')

⊕(g1::Gaussian{P}, g2::Gaussian{P}) where {P <: NamedTuple{(:μ, :Σ)}} = Gaussian{P}(g1.μ + g2.μ, g1.Σ + g2.Σ)
⊕(x, g::Gaussian) = x + g
⊕(g::Gaussian, x) = g + x
