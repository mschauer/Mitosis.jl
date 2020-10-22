using Mitosis, LinearAlgebra, Test, Statistics
m = [1.0, 0.5]
K = 10000
Q = Matrix(1.0I, 2, 2)
G = Gaussian(μ=m, Σ=Q)

@test m === mean(G)
@test Q === cov(G)

@test 10/sqrt(K) > norm(m - mean(rand(G) for x in 1:K))
@test 10/sqrt(K) > norm(Q - cov([rand(G) for x in 1:K]))
