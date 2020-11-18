using Revise
using Mitosis
using Random, Test, LinearAlgebra, Statistics
using Distributions, StatsBase
using Mitosis: outer
const d = 4

# Forward model and data generation: multivariate Wiener
# X = x0 + μ*t + σ*W(t)

μ = randn(d)
σ = randn(d,d)
Σ = outer(σ)
n = 500
t = [0.0; sort(1000*rand(n-1))]
n = length(t)
function bm(x, s, μ, σ)
    y = float(x)
    ys = [y]
    for i in 1:length(s)-1
        Δ = s[i+1] - s[i]
        k = kernel(Gaussian; μ=AffineMap(I(d), Δ*μ), Σ=ConstantMap(Σ*Δ))
        y = rand(k(y))
        push!(ys, y)
    end
    ys
end
x0 = zeros(d)
x = bm(x0, t, μ, σ)

# Conjugate analysis of increments

using Distributions
E(i, d) = Vector(I(d)[:,i])

function gibbs(t, x, (μ0, Σ0), πμ, πΣ, iterations)
    n = length(t)
    d = length(μ0)
    μ = μ0
    Σ = Σ0
    out = [(μ, σ)]
    for iter in 1:iterations
        # conjugate μ
        p = πμ
        for i in n-1:-1:1
            Δ = t[i+1] - t[i]
            r = Mitosis.Regression(μ, [ConstantMap(Δ*E(i, d)) for i in 1:d], identity)
            k = kernel(Gaussian; μ=r, Σ=ConstantMap(Σ*Δ))
            con = Mitosis.conjugate(k, x[i])
            _, pi = backward(BF(), con, x[i+1])
            _, p = fuse(p, pi)
        end
        μ = rand(convert(Gaussian{(:μ,:Σ)}, p))

        α = params(πΣ)[1]
        β = Matrix(params(πΣ)[2])

        for i in n-1:-1:1
            Δ = t[i+1] - t[i]
            α = α + 1/2
            k = kernel(Gaussian; μ=AffineMap(I(d), Δ*μ), Σ=ConstantMap(Σ*Δ))
            β = β + Mitosis.outer(x[i+1] .- mean(k(x[i])))/(2Δ)
        end
        Σ = rand(InverseWishart(α, β))

        push!(out, (μ, Σ))
    end
    out
end


F0, Γ0 = zeros(d), zeros(d,d)
πμ = Gaussian{(:F,:Γ)}(F0, Γ0)
πΣ = InverseWishart(d+4, zeros(d,d) + 20I)
var(πΣ)
Σ0 = mean(πΣ)
chain = gibbs(t, x, (F0, Σ0), πμ, πΣ, 2000)
μs = first.(chain)
Σs = last.(chain)

display([μ mean(μs)])
display([vec(Σ) vec(mean(Σs))])
