import Statistics: mean, cov
import Random.rand
struct WGaussian{P,T} <: AbstractMeasure
    par::NamedTuple{P,T}
end
WGaussian{P}(nt::NamedTuple{P,T}) where {P,T} = WGaussian{P,T}(nt)
WGaussian{P}(args...) where {P} = WGaussian(NamedTuple{P}(args))
WGaussian(;args...) = WGaussian(args.data)
WGaussian{P}(;args...) where {P} = WGaussian{P}(args.data)

# the following propagates uncertainty if `μ` is `Gaussian`
WGaussian(par::NamedTuple{(:μ,:Σ,:c),Tuple{T,S,U}}) where {T<:WGaussian,S,U} = WGaussian((;μ=mean(par.μ), Σ=par.Σ +cov(par.μ), c=par.c + par.μ.c))
WGaussian{P}(par::NamedTuple{(:μ,:Σ,:c),Tuple{T,S,U}}) where {P,T<:WGaussian,S,U} = WGaussian{P}(mean(par.μ), par.Σ + cov(par.μ), par.c + par.μ.c)

Base.getproperty(p::WGaussian, s::Symbol) = getproperty(getfield(p, :par), s)

const WGaussianOrNdTuple{P} = Union{WGaussian{P},NamedTuple{P}}

Base.keys(p::WGaussian{P}) where {P} = P
params(p::WGaussian) = getfield(p, :par)

dim(p::WGaussian{(:F, :Γ, :c)}) = length(p.F)

mean(p::WGaussian{(:μ, :Σ, :c)}) = p.μ
cov(p::WGaussian{(:μ, :Σ, :c)}) = p.Σ

mean(p::WGaussian{(:F, :Γ, :c)}) = p.Γ\p.F
cov(p::WGaussian{(:F, :Γ, :c)}) = inv(p.Γ)
moment1(p::WGaussian{(:F, :Γ, :c)}) = p.c*p.Γ\p.F

Base.isapprox(p1::WGaussian, p2::WGaussian; kwargs...) =
    all(isapprox.(Tuple(params(p1)), Tuple(params(p2)); kwargs...))

function MeasureTheory.logdensityof(p::WGaussian{(:F,:Γ,:c)}, x)
    C = cholesky_(sym(p.Γ))
    p.c - x'*p.Γ*x/2 + x'*p.F - p.F'*(C\p.F)/2  + logdet(C)/2 - dim(p)*log(2pi)/2
end
densityof(p::WGaussian, x) = exp(logdensityof(p, x))

StatsBase.rand(p::WGaussian) = rand(Random.GLOBAL_RNG, p)
StatsBase.rand(RNG::AbstractRNG, p::WGaussian{(:μ,:Σ,:c)}) = weighted(unwhiten(Gaussian{(:μ,:Σ)}(p.μ, p.Σ), randn!(RNG, zero(mean(p)))), p.c)
weighted(p::WGaussian{(:μ, :Σ, :c)}, ll) = WGaussian{(:μ, :Σ, :c)}(p.μ, p.Σ, p.c + ll)

function Base.convert(::Type{WGaussian{(:F,:Γ,:c)}}, p::WGaussian{(:μ,:Σ,:c)})
    Γ = inv(p.Σ)
    return WGaussian{(:F,:Γ,:c)}(Γ*p.μ, Γ, p.c)
end

function Base.convert(::Type{WGaussian{(:μ,:Σ,:c)}}, p::WGaussian{(:F,:Γ,:c)})
    Σ = inv(p.Γ)
    return WGaussian{(:μ,:Σ,:c)}(Σ*p.F, Σ, p.c)
end
Base.convert(::Type{WGaussian{(:μ,:Σ,:c)}}, p::Leaf) = convert(WGaussian{(:μ,:Σ,:c)}, p[])
Base.convert(::Type{WGaussian{(:F,:Γ,:c)}}, p::Leaf) = convert(WGaussian{(:F,:Γ,:c)}, p[])
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

MeasureTheory.logdensityof(p::WGaussian, x) = -(sqmahal(p,x) + _logdet(p, dim(p)) + dim(p)*log(2pi))/2
MeasureTheory.densityof(p::WGaussian, x) = exp(logpdf(p, x))

=#
