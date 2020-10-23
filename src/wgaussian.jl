import Statistics: mean, cov
import Random.rand
struct WGaussian{P,T} <: AbstractMeasure
    par::NamedTuple{P,T}
end
WGaussian{P}(nt::NamedTuple{P,T}) where {P,T} = WGaussian{P,T}(nt)
WGaussian{P}(args...) where {P} = WGaussian(NamedTuple{P}(args))
WGaussian(;args...) = WGaussian(args.data)
WGaussian{P}(;args...) where {P} = WGaussian{P}(args.data)


Base.getproperty(p::WGaussian, s::Symbol) = getproperty(getfield(p, :par), s)

const WGaussianOrNdTuple{P} = Union{WGaussian{P},NamedTuple{P}}

Base.keys(p::WGaussian{P}) where {P} = P

dim(p::WGaussian{(:F, :Γ, :c)}) = length(p.F)


function MeasureTheory.logdensity(p::WGaussian{(:F,:Γ,:c)}, x)
    C = cholesky(p.Γ)
    p.c - x'*p.Γ*x/2 + x'*p.F - p.F'*(C\p.F)/2  + logdet(C)/2 - dim(p)*log(2pi)/2
end
MeasureTheory.density(p::WGaussian, x) = exp(logpdf(p, x))

#=
Base.:(==)(p1::WGaussian, p2::WGaussian) = mean(p1) == mean(p2) && cov(p1) == cov(p2)
Base.isapprox(p1::WGaussian, p2::WGaussian; kwargs...) =
    isapprox(mean(p1), mean(p2); kwargs...) && isapprox(cov(p1), cov(p2); kwargs...)

mean(p::WGaussian{(:μ, :Σ)}) = p.μ
cov(p::WGaussian{(:μ, :Σ)})  = p.Σ
meancov(p) = mean(p), cov(p)

precision(p::WGaussian{(:μ, :Σ)}) = inv(p.Σ)

mean(p::WGaussian{(:F, :Γ)}) = Γ\p.F
cov(p::WGaussian{(:F, :Γ)}) = inv(p.Γ)
precision(p::WGaussian{(:F, :Γ)}) = p.Γ

dim(p::WGaussian) = length(mean(p))
whiten(p::WGaussian{(:μ, :Σ)}, x) = cholesky(p.Σ).U'\(x - p.μ)
unwhiten(p::WGaussian{(:μ, :Σ)}, z) = cholesky(p.Σ).U'*z + p.μ
sqmahal(p::WGaussian, x) = norm_sqr(whiten(p, x))

rand(p::WGaussian) = rand(Random.GLOBAL_RNG, p)
rand(RNG::AbstractRNG, p::WGaussian) = unwhiten(p, randn!(RNG, zero(mean(p))))

MeasureTheory.logdensity(p::WGaussian, x) = -(sqmahal(p,x) + _logdet(p, dim(p)) + dim(p)*log(2pi))/2
MeasureTheory.density(p::WGaussian, x) = exp(logpdf(p, x))

=#
