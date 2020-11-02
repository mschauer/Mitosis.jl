struct Regression{S,T,U}
    β::S
    f::T
    f0::U
end
(r::Regression)(x) = r.f0(x) + sum(r.β[i]*r.f[i](x) for i in eachindex(r.β))

function conjugate(k, x)
    r = k.ops.μ::Regression
    Q = k.ops.Σ(x)
    w = [f(x) for f in r.f]
    w0_ = r.f0(x)
    w0 = size(w0_) == () ? [w0_] : w
    W = [w[j][i] for i in eachindex(w[1]), j in eachindex(w)]
    kernel(Gaussian; μ=AffineMap(W, w0), Σ=ConstantMap(Q))
end
# X ~ N(β'*F, Q)
