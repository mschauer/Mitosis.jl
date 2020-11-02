struct Regression{S,T,U}
    β::S
    f::T
    f0::U
end
(r::Regression)(x) = r.f0(x) + sum(r.β[i]*r.f[i](x) for i in eachindex(r.β))

# X ~ N(β'*F, Q)
