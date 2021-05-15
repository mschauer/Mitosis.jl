using Mitosis, LinearAlgebra, Test, Statistics
m = [1.0, 0.5]
K = 10000
Q = [1.0 0.2; 0.2 1.0]
G = Gaussian(μ=m, Σ=Q)

@test G === Gaussian{keys(G)}(m, Q)


p = WGaussian(μ=m, Σ=Q, c=0.0)
p2 = convert(WGaussian{(:μ,:Σ,:c)}, convert(WGaussian{(:F,:Γ,:c)}, p))
@test p ≈ p2

p = Gaussian(μ=m, Σ=Q)
p2 = convert(Gaussian{(:F,:Γ)}, p)
p3 = convert(Gaussian{(:μ,:Σ)}, p2)
@test p ≈ p3

@testset "Gauss" begin
    for G in [p, p2]
        @test m ≈ mean(G)
        @test Q ≈ cov(G)

        @test 10/sqrt(K) > norm(m - mean(rand(G) for x in 1:K))
        @test 10/sqrt(K) > norm(Q - cov([rand(G) for x in 1:K]))
    end
end