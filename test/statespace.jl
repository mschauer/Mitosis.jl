#using Revise
using Mitosis


using Mitosis: kernel, correct, Kernel, Gaussian, ConstantMap, AffineMap, GaussKernel
using Random, Test, LinearAlgebra, Statistics

Random.seed!(1)
K = 10000 # repetitions

# State space system
#
# x[0] ∼ N(x0, P0)
# x[k] = Φx[k−1] + w[k],    w[k] ∼ N(0, Q)
# y[k] = Hx[k] + v[k],    v[k] ∼ N(0, R)

x0 = [1., 0.]
P0 = Matrix(1.0I, 2, 2)

Φ = [0.8 0.5; -0.1 0.8]
β = [0.1, 0.2]
Q = [0.2 0.0; 0.0 1.0]

yshadow = [0.0]
H = [1.0 0.0]
R = Matrix(1.0I, 1, 1)
INoise = Gaussian(μ=zero(x0), Σ=Q)
Noise = Gaussian(μ=zero(yshadow),Σ=R)

@test AffineMap(Φ, β)(x0) == Φ*x0 + β
@test ConstantMap(β)(x0) == β

transition = kernel(Gaussian; μ=AffineMap(Φ, β), Σ=ConstantMap(Q))
transition2 = kernel(Gaussian; μ=AffineMap(Φ, 0β), Σ=ConstantMap(Q))

observation = kernel(Gaussian; μ=AffineMap(H, yshadow), Σ=ConstantMap(R))

@test 10/sqrt(K) > norm(Φ*x0 + β - mean(rand(transition(x0)) for x in 1:K))

@test transition.ops isa NamedTuple{(:μ, :Σ)}

y0 = rand(observation(x0))

x1 = rand(transition(x0))
y1 = rand(observation(x1))

x2 = rand(transition(x1))
y2 = rand(observation(x2))

p0 = Gaussian(μ=x0, Σ=P0)

@test mean(AffineMap(Φ, β)(p0)) == Φ*x0 + β
@test cov(AffineMap(Φ, β)(p0)) == Φ*P0*Φ'



q0 = observation(p0)
@test cov(q0) == H*P0*H' + R
p1 = transition2(p0)
@test cov(p1) == Φ*P0*Φ' + Q

q1 = observation(p1)
p2 = transition2(p1)
@test cov(p2) == Φ*(Φ*P0*Φ' + Q)*Φ' + Q
q2 = observation(p2)

flat(x) = collect(Iterators.flatten(x))


# Check that this is all correct:

# Write down joint distribution of x's and y's by hand... (yes I am fine ;-))
# Define mean and covariance of the flattened vector of states and observations [x0 x1 x2 y0 y1 y2]
μ_ = [1.0, 0.0, 0.8, -0.1, 0.59, -0.16, 1.0, 0.8, 0.59]
μ = [ x0; Φ*x0; Φ*Φ*x0; H*x0; H*Φ*x0; H*Φ*Φ*x0; ]
@test μ_ ≈ μ


P0L = cholesky(P0).L
QL = cholesky(Q).L
RL = cholesky(R).L
Z(m,n) = zeros(m,n)
# Lower cholesky factor from "innovation form"
L = [   [     P0L  Z(2,2) Z(2,2)  Z(2,1)  Z(2,1)  Z(2,1)]
        [    Φ*P0L     QL Z(2,2)  Z(2,1)  Z(2,1)  Z(2,1)]
        [  Φ*Φ*P0L   Φ*QL     QL  Z(2,1)  Z(2,1)  Z(2,1)]
        [    H*P0L Z(1,2) Z(1,2)     RL  0I  0I]
        [  H*Φ*P0L   H*QL Z(1,2)     0I  RL  0I]
        [H*Φ*Φ*P0L H*Φ*QL   H*QL     0I  0I  RL]
]

Σ = L*L'

@testset "Uncertainty propagation" begin
        @test μ ≈ flat(mean.([p0, p1, p2, q0, q1, q2]))
        @test diag(Σ) ≈ flat(diag.(cov.([p0, p1, p2, q0, q1, q2])))
end

# Compute the conditional distribution of vector `x0` given data.
π0 = Mitosis.conditional(Gaussian(;μ=μ, Σ=Σ), 1:2, 7:9, vcat(y0, y1, y2))





# Compute the conditional distribution of vector `x0` given data.
π2 = Mitosis.conditional(Gaussian(;μ=μ, Σ=Σ), 5:6, 7:9, vcat(y0, y1, y2))

@testset "Kalman filter" begin
        p0f = correct(p0, observation, y0)[1]
        p1p = transition2(p0f)
        p1f = correct(p1p, observation, y1)[1]
        p2p = transition2(p1f)
        p2f = correct(p2p, observation, y2)[1]

        @test mean(p2f) ≈ mean(π2)
        @test cov(p2f) ≈ cov(π2)
end
