inner(x) = dot(x,x)
outer(x) = x*x'
sym(x) = Symmetric(x)
sym(x::Number) = x
_logdet(Σ, d::Integer) = LinearAlgebra.logdet(Σ)
lchol(Σ) = cholesky(sym(Σ)).U'
lchol(Σ::Number) = sqrt(Σ)
cholesky_(x) = cholesky(x)
cholesky_(x::Number) = x

struct Zero <: AbstractVector{Bool}
end
Base.:+(x::AbstractVector, z::Zero) = x
Base.:+(z::Zero, x::AbstractVector) = x

Base.:*(x::AbstractMatrix, z::Zero) = z
Base.:-(x::AbstractVector, z::Zero) = x
flat(x) = collect(Iterators.flatten(x))
