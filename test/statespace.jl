if !@isdefined TEST
    using Revise
end
using Mitosis


using Mitosis: meancov, flat
using Random, Test, LinearAlgebra, Statistics

Random.seed!(1)
K = 10000 # repetitions

# State space system
#
# x[0] ∼ N(x0, P0)
# x[k] = Φx[k−1] + w[k],    w[k] ∼ N(0, Q)
# y[k] = Hx[k] + v[k],    v[k] ∼ N(0, R)

ξ0 = [1., 0.]
P0 = Matrix(1.0I, 2, 2)
prior = Gaussian{(:μ,:Σ)}(ξ0, P0)

Φ = [0.8 0.5; -0.1 0.8]
β = [0.1, 0.2]
Q = [0.2 0.0; 0.0 1.0]

yshadow = [0.0]
H = [1.0 0.0]
R = Matrix(1.0I, 1, 1)
INoise = Gaussian(μ=zero(ξ0), Σ=Q)
Noise = Gaussian(μ=zero(yshadow),Σ=R)

@test AffineMap(Φ, β)(ξ0) == Φ*ξ0 + β
@test ConstantMap(β)(ξ0) == β


priortransition = kernel(Gaussian; μ=LinearMap(I(2)), Σ=ConstantMap(P0))
@test priortransition(ξ0) ≈ prior

transition = kernel(Gaussian; μ=AffineMap(Φ, β), Σ=ConstantMap(Q))
transition2 = kernel(Gaussian; μ=AffineMap(Φ, β), Σ=ConstantMap(Q))
transition1 = kernel(Gaussian; μ=LinearMap(Φ), Σ=ConstantMap(Q))
transition1 = transition2

observation = kernel(Gaussian; μ=LinearMap(H), Σ=ConstantMap(R))

@test 10/sqrt(K) > norm(Φ*ξ0 + β - mean(rand(transition(ξ0)) for x in 1:K))

@test transition.ops isa NamedTuple{(:μ, :Σ)}

x0 = rand(prior)
y0 = rand(observation(x0))

x1 = rand(transition(x0))
y1 = rand(observation(x1))

x2 = rand(transition(x1))
y2 = rand(observation(x2))

p0 = Gaussian(μ=ξ0, Σ=P0)


@test mean(AffineMap(Φ, β)(p0)) == Φ*ξ0 + β
@test cov(AffineMap(Φ, β)(p0)) == Φ*P0*Φ'



q0 = observation(p0)
@test cov(q0) == H*P0*H' + R
p1 = transition2(p0)
@test cov(p1) == Φ*P0*Φ' + Q

q1 = observation(p1)
p2 = transition2(p1)
@test cov(p2) == Φ*(Φ*P0*Φ' + Q)*Φ' + Q
q2 = observation(p2)


# Check that this is all correct:

# Write down joint distribution of x's and y's by hand... (yes I am fine ;-))
# Define mean and covariance of the flattened vector of states and observations [x0 x1 x2 y0 y1 y2]
#μ_ = [1.0, 0.0, 0.8, -0.1, 0.59, -0.16, 1.0, 0.8, 0.59]
μ_ = [1.0, 0.0, 0.9, 0.1, 0.87, 0.19, 1.0, 0.9, 0.87] # with beta
μ = [ ξ0; Φ*ξ0 + β; Φ*(Φ*ξ0 + β)+β; H*ξ0; H*(Φ*ξ0 + β); H*(Φ*(Φ*ξ0 + β)+β); ]
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


@testset "right' linear gaussian case" begin
    p0 = right′(BF(), transition1, p1)[2]

    ν, P = meancov(p0)

    @test mean(transition1(ν)) ≈ mean(p1)
    @test Φ*cov(p0)*Φ' ≈ (cov(p1) + Q)



    H = inv(cov(p1))
    q1 = WGaussian(F=H*mean(p1), Γ=H, c=0.0)
    @test logdensity(q1, x0) ≈ logdensity(p1, x0)

    q0 = right′(BFFG(), transition1, q1)[2]

    ν0 = q0.Γ\q0.F
    P0 = inv(q0.Γ)
    @test mean(transition1(ν0)) ≈ mean(p1)
    @test Φ*P0*Φ' ≈ (cov(p1) + Q)

    pp = transition1(x0)⊕Gaussian(μ=0*p1.μ, Σ=p1.Σ)
    @test logdensity(q0, x0) ≈ logdensity(pp, p1.μ)

    @test 0.0 ≈ -logdet(pp.Σ)/2 - (q0.c + logdet(q0.Γ)/2) atol=1e-10
end

#=
# Next we consider the following model, in Soss notation
m = @model ξ0 begin
           x0 ~ MvNormal(ξ0, P0) # priortransition
           y0 ~ MvNormal(H*x0, R) # observation
           x1 ~ MvNormal(Φ*x0, Q) # transition2
           y1 ~ MvNormal(H*x1, R) # observation
           x2 ~ MvNormal(Φ*x1, Q) # transition2
           return y0, y1, x2
end
=#

@testset "Backward filter with fusion" begin
    # forward model
    x0 = rand(priortransition(ξ0))
    y0 = rand(observation(x0))
    x1 = rand(transition(x0))
    y1 = rand(observation(x1))
    x2 = rand(transition(x1))

    # run backward filter
    m1a, p1a = backwardfilter(observation, y1; unfused=true)
    m1b, p1b = backwardfilter(transition2, x2; unfused=true)
    m1, p1 = fuse(p1a, p1b)
    m0a, p0a = backwardfilter(observation, y0; unfused=true)
    m0b, p0b = backwardfilter(transition2, p1; unfused=true)
    m0, p0 = fuse(p0a, p0b)
    m, evi = backwardfilter(priortransition, p0)

    # as byproduct this just computed the model evidence as function
    # of ξ0
    @test logdensity(evi, ξ0) ≈ logdensity(Gaussian(;μ=μ[5:8], Σ=Σ[5:8,5:8]), vcat(x2, y0, y1))


    # Alternative: p0_ is the conditional distribution of x0
    prior_ = WGaussian{(:F,:Γ,:c)}(P0\ξ0, inv(P0), 0.0)
    m0_, p0_ = fuse(p0a, p0b, prior_)
    π0 = Mitosis.conditional(Gaussian(;μ=μ, Σ=Σ), 1:2, 5:8, vcat(x2, y0, y1))
    @test mean(π0) ≈ mean(p0_)
    @test cov(π0) ≈ cov(p0_)



    # alternative with fusion
    _, evi_ = backwardfilter(priortransition, p0; unfused=true)
    _, evi2 = fuse(evi_)
    @test evi2 ≈ evi

    # run forward marginal smoother
    kᵒ = left′(BFFG(), priortransition, m)
    p0ᵒ = kᵒ(ξ0)
    # skip copy
    k0ᵒ = left′(BFFG(), transition2, m0b)
    p1ᵒ = k0ᵒ(p0ᵒ)




    # first step gives the conditional marginal of x0
    @test mean(π0) ≈ mean(p0ᵒ)
    @test cov(π0) ≈ cov(p0ᵒ)

    # second step gives the conditional marginal of x1
    π1 = Mitosis.conditional(Gaussian(;μ=μ, Σ=Σ), 3:4, 5:8, vcat(x2, y0, y1))
    @test mean(π1) ≈ mean(p1ᵒ)
    @test cov(π1) ≈ cov(p1ᵒ)


    # some more tests
    @test logdensity(backwardfilter(transition2, x2)[2], x1) ≈ logdensity(transition2(x1), x2)
    @test logdensity(fuse(backwardfilter(transition2, x2; unfused=true)[2])[2], x1) ≈ logdensity(transition2(x1), x2)

    _, q = fuse(backwardfilter(transition2, x2; unfused=true)[2], backwardfilter(transition2, x2; unfused=true)[2])
    @test logdensity(q, x1) ≈ 2logdensity(transition2(x1), x2)
    @test logdensity(p1, x1) ≈ logdensity(observation(x1), y1) + logdensity(transition2(x1), x2)

end
