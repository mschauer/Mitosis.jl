using Mitosis, LinearAlgebra, Test, Statistics
m = [1.0, 0.5]
K = 10000
Q = Matrix(1.0I, 2, 2)
G = Gaussian(μ=m, Σ=Q)

@testset "Gauss" begin
    @test G === Gaussian{keys(G)}(m, Q)

    @test m === mean(G)
    @test Q === cov(G)

    @test 10/sqrt(K) > norm(m - mean(rand(G) for x in 1:K))
    @test 10/sqrt(K) > norm(Q - cov([rand(G) for x in 1:K]))
end

p = WGaussian(μ=m, Σ=Q, c=0.0)
p2 = convert(WGaussian{(:μ,:Σ,:c)}, convert(WGaussian{(:F,:Γ,:c)}, p))
@test p ≈ p2
