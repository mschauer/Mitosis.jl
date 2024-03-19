if !@isdefined TEST
    using Revise
end
using Mitosis
using Random, Test, LinearAlgebra, Statistics
using Soss
using StatsBase
DIAG = true
if DIAG
    using MCMCChains
end

N = 50_000 # samples


# Define some vectors and matrices
ξ0 = [1., 0.]
x = [1.2, 0.1]
P0 = Matrix(1.0I, 2, 2)

Φ = [0.8 0.5; -0.1 0.8]
β = [0.1, 0.2]
Q = [0.2 0.0; 0.0 1.0]

H = [1.0 0.0]
R = Matrix(1.0I, 1, 1)

# and a nonlinear vector function
f(x) = [0.8 0.5; -0.1 0.8]*[atan(x[1]), atan(x[2])] + [0.1, 0.2]


# We define the equivalent of the Soss model
(m = @model ξ0, H, f, P0, R, Q begin
    a ~ MvNormal(ξ0, P0) # priortransition
    b ~ MvNormal(H*a, R) # partial observation
    c ~ MvNormal(f(a), Q) # nonlineartransition
    d ~ MvNormal(H*c, R) # partial observation
    e ~ MvNormal(c, P0) # full observation
    # return b, d, e
end)

# Define some transition kernels.

# We use AffineMap, LinearMap and ConstantMap
# For example
@test AffineMap(Φ, β)(x) == Φ*x + β

# Prior
# a ~ N(ξ0, P0)
k_0a = Mitosis.kernel(Gaussian; μ=LinearMap(I(2)), Σ=ConstantMap(P0))

# Partial observation
# b ~ N(H*a, R)
k_ab = Mitosis.kernel(Gaussian; μ=LinearMap(H), Σ=ConstantMap(R))

# Nonlinear transition and linear approximation
# c ~ N(f(a), Q)
k_ac = Mitosis.kernel(Gaussian; μ=f, Σ=ConstantMap(Q))
k_ac_approx = Mitosis.kernel(Gaussian; μ=AffineMap(Φ, β), Σ=ConstantMap(Q))


# d ~ N(H*c, R)
k_cd = Mitosis.kernel(Gaussian; μ=LinearMap(H), Σ=ConstantMap(R))

# e ~ N(c, P0)
k_ce = Mitosis.kernel(Gaussian; μ=LinearMap(I(2)), Σ=ConstantMap(P0))


# And a deterministic copy kernel
cp2 = Mitosis.Copy{2}() # copy kernel
@test cp2(1.1) == (1.1, 1.1)

# Forward sample a Bayes net # rand(m(ξ0=ξ0))
a = rand(k_0a(ξ0))
b = rand(k_ab(a))
c = rand(k_ac(a)) # forward model with nonlinear transition
d = rand(k_cd(c))
e = rand(k_ce(c))


# We actually want to write down the
# forward sample with explicit copies
# so every state is only used as input of a single kernel
a = rand(k_0a(ξ0))
a1, a2 = cp2(a)
b = rand(k_ab(a1))
c = rand(k_ac(a2)) # forward model with nonlinear transition
c1, c2 = cp2(c)
d = rand(k_cd(c1))
e = rand(k_ce(c2))

# sample posterior with dynamicHMC via Soss for comparison
# monkey patch for xform
Soss.xform(d::MvNormal, _data) = Soss.as(Array, Soss.asℝ, size(d))
@time posterior = dynamicHMC(m(ξ0=ξ0, H=H, f=f, P0=P0, R=R, Q=Q), (b=b, d=d, e=e,), N)


# observations are b d e (all leaves)
mc2, pc2 = backward(BFFG(), k_ce, e; unfused=true) # child of copy
mc1, pc1 = backward(BFFG(), k_cd, d; unfused=true) # child of copy
mc, pc = backward(BFFG(), cp2, pc1, pc2) # copy
ma2, pa2 = backward(BFFG(), k_ac_approx, pc; unfused=true) # # child of copy
ma1, pa1 = backward(BFFG(), k_ab, b; unfused=true) # child of copy
ma, pa = backward(BFFG(), cp2, pa1, pa2) # copy
m0, p0 = backward(BFFG(), k_0a, pa) # not a child of a copy

samples = []
lls = []
@time for i in 1:N
    a = rand(forward(BFFG(), k_0a, m0, weighted(ξ0)))
    a1, a2 = rand(forward(BFFG(), cp2, ma, a))
    b = rand(forward(BFFG(), k_ab, ma1, a1))
    c = rand(forward(BFFG(), k_ac, ma2, a2)) # forward model with nonlinear transition
    c1, c2 = rand(forward(BFFG(), cp2, mc, c))
    d = rand(forward(BFFG(), k_cd, mc1, c1))
    e = rand(forward(BFFG(), k_ce, mc2, c2))

    # Sample and weight (sum over the weights of the leafs.)
    push!(samples, (a=a.x,b=b.x,c=c.x,d=d.x,e=e.x))
    push!(lls, b.ll + d.ll + e.ll)

end


# Importance weights
w = exp.(lls)/mean(exp.(lls))

# Estimate Mitosis
â1 = mean(getfield.(samples, :a).*w)
v1 = mean(Mitosis.outer.(getfield.(samples, :a).-Ref(â1)).*w)

# Estimate Soss
â2 = mean(getfield.(posterior, :a))
v2 = cov(getfield.(posterior, :a))


if DIAG
    # Kish's Effective Sample Size
    ess1 = sum(w).^2/sum(w.^2)
    ess2 = MCMCChains.ess(Chains(reshape(first.(getfield.(posterior, :a)), :,1,1)))
end


[â1 â2]
