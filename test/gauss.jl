using Mitosis
G = Gaussian([1.0], Matrix(1.0I, 2, 2))
m, K = G
@test m === mean(G)
@test K === cov(G)
