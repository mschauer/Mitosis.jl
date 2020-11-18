struct Regression{S,T,U}
    β::S
    f::T
    f0::U
end
(r::Regression)(x) = r.f0(x) + sum(r.β[i]*r.f[i](x) for i in eachindex(r.β))

function conjugate(k, u)
    r = k.ops.μ::Mitosis.Regression
    Q = k.ops.Σ(u)
    w = [f(u) for f in r.f]
    w0_ = r.f0(u)
    w0 = size(w0_) == () ? [w0_] : w0_
    W = [w[j][i] for i in eachindex(w[1]), j in eachindex(w)]
    kernel(Gaussian; μ=AffineMap(W, w0), Σ=ConstantMap(Q))
end


function conjugate2(k, x)
    y - mu(x)
    r = k.ops.Σ::Regression
    μ = k.ops.μ(x)
    w = [f(x) for f in r.f]
    @assert iszero(r.f0(x))
    W = [w[j][i] for i in eachindex(w[1]), j in eachindex(w)]
    kernel(Gamma; μ=AffineMap(W, w0), Σ=ConstantMap(Q))
end
