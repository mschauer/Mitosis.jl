if !@isdefined TEST
    using Revise
end
using Mitosis
using Random, Test, LinearAlgebra, Statistics
Random.seed!(1)

# Forward model
β = [0.1, 0.2]


# Prior (as kernel of prior mean)
I2 = Matrix(1.0I(2))
μ0 = zeros(2)
σ2 = 8.0 # noise
V0 = 10*I2
Σ0 = σ2*V0 # prior

prior = kernel(Gaussian; μ=ConstantMap(μ0), Σ=ConstantMap(Σ0))

# Data
x = [18.25 19.75 16.5 18.25 19.50 16.25 17.25 19.00 16.25 17.50][:]
y = [36 42 33 39 43 34 37 41 27 30][:]
n = length(x)

# Model
X = [x ones(n)]
Σ = Diagonal(σ2*ones(n))
model = kernel(Gaussian; μ=LinearMap(X), Σ=ConstantMap(Σ))

# backward pass
m2, p2 = right′(BF(), model, y)
m1, p1 = right′(BF(), prior, p2)

posterior = left′(BF(), prior, m1)()

@show mean(posterior), cov(posterior)

β̂ = X\y
@test β̂ ≈ mean(p2)

@test mean(posterior) ≈ inv(X'*X + inv(V0))*X'*y
@test cov(posterior) ≈ σ2*inv(X'*X + inv(V0))
