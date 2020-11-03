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
μ0 = zero(β)
σ2 = 8.0
V0 = 10*I2
Σ0 = σ2*V0
prior = kernel(Gaussian; μ=LinearMap(I2), Σ=ConstantMap(Σ0))
p0 = convert(Gaussian{(:F,:Γ)}, Gaussian(μ=μ0, Σ=Σ0))

# Data
x = [18.25 19.75 16.5 18.25 19.50 16.25 17.25 19.00 16.25 17.50][:]
y = [36 42 33 39 43 34 37 41 27 30][:]
n = length(x)
Σ = fill(σ2, 1,1)
# Model (as kernel)
ks = [kernel(Gaussian; μ=LinearMap([x[i] 1.0]), Σ=ConstantMap(Σ)) for i in eachindex(x)]


# backward pass
z = [right′(BF(), ks[i], [y[i]]) for i in 1:n]
_, ps = first.(z), last.(z)
_, p = right′(BF(), Copy{Any}(), ps...)
m, evidence = right′(BF(), prior, p)

posterior = left′(BF(), prior, m)(μ0)

@show mean(posterior), cov(posterior)


X = [x ones(n)]

β̂ = X\y
@test β̂ ≈ mean(p)

@test mean(posterior) ≈ inv(X'*X + inv(V0))*X'*y
@test cov(posterior) ≈ σ2*inv(X'*X + inv(V0))
