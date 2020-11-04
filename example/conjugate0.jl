if !@isdefined TEST
    using Revise
end
using Mitosis
using Random, Test, LinearAlgebra, Statistics
using BayesianLinearRegressors
Random.seed!(1)

# Forward model
β = [0.1, 0.2]

r1 = Mitosis.Regression(β, (sin,cos), zero)
r2 = Mitosis.Regression(β, (cos,cos), zero)
k1 = kernel(Gaussian, μ=r1, Σ=ConstantMap(1.0))
k2 = kernel(Gaussian, μ=r2, Σ=ConstantMap(2.0))

# Data
x0 = rand()
x1 = rand(k1(x0))
x2 = rand(k2(x1))

# Prior
mw, Λw = zeros(2), Matrix(1.0I(2))
p = Gaussian{(:F,:Γ)}(mw, Λw)

# Conjugacy
con1 = Mitosis.conjugate(k1, x0)
con2 = Mitosis.conjugate(k2, x1)
m1, p1 = backward(BF(), con1, [x1]);
m2, p2 = backward(BF(), con2, [x2]);
m, p = fuse(p, p1, p2)

# Check with BayesianLinearRegressor
brprior = BayesianLinearRegressor(mw, Λw)
w1 = con1.ops.μ.B
w2 = con2.ops.μ.B
q1 = con1.ops.Σ.x
q2 = con2.ops.Σ.x

br = brprior([w1'  w2'], diagm([q1,q2]))
post = posterior(br, [x1, x2])

@test p.Γ ≈ post.Λw
@test mean(p) ≈ post.mw
