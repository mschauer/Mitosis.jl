abstract type AbstractMeasure end

struct Kernel{T, NT}
    ops::NT
end
mapcall(t) = map(func -> func(), t)
mapcall(t, arg) = map(func -> func(arg), t)
(k::Kernel{M,<:NamedTuple})() where {M} = M(mapcall(k.ops))
(k::Kernel{M,NT})(arg) where {M,NT<:NamedTuple} = M(mapcall(k.ops, arg))


Kernel{T}(args::NT) where {T,NT} = Kernel{T,NT}(args)
kernel(::Type{T}; kwargs...) where {T} = Kernel{T}(NamedTuple(kwargs))
struct Dirac{T} <: AbstractMeasure
    x::T
end
rand(d::Dirac) = d.x
rand(_::AbstractRNG, d::Dirac) = d.x
