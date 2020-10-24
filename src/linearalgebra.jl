inner(x) = dot(x,x)
outer(x) = x*x'
sym(x) = Symmetric(x)
_logdet(Σ, d::Integer) = LinearAlgebra.logdet(Σ)
lchol(Σ) = cholesky(sym(Σ)).U'

struct Zero <: AbstractVector{Bool}
end
Base.:+(x::Vector, z::Zero) = x
Base.:-(x::Vector, z::Zero) = x
flat(x) = collect(Iterators.flatten(x))
