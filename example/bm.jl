# Fit a Brownian motion with drift to and scale to
# Georgia election data:
# fraction of total count of votes vs difference in votes

data = [0.9222672139181572 103896.10389610389
    0.9267802009311443 87662.33766233765
    0.9277217593727027 85714.28571428571
    0.9285683655966674 83116.88311688312
    0.9299797843665768 79220.77922077922
    0.9310169076206811 78571.42857142857
    0.9329974271012007 77922.07792207792
    0.9332804459691252 77922.07792207792
    0.9364788042146532 68181.81818181818
    0.9368567753001715 68831.16883116882
    0.9372335211957853 68181.81818181818
    0.9426035285469248 60389.61038961039
    0.9435438617985787 57142.85714285713
    0.9438268806665033 57142.85714285713
    0.9471195785346729 47402.597402597385
    0.9484409458466062 48051.94805194806
    0.9526721391815731 33116.88311688311
    0.9532369517275178 31818.18181818181
    0.9578473413379074 18831.16883116882]
t = data[:, 1]
x = data[:, 2]

using Mitosis
using Random, Test, LinearAlgebra, Statistics
using Distributions

function gibbs(t, x, iterations)
    # prior μ
    mw, Λw = zeros(1), Matrix(1e-12I(1))
    # prior σ2
    α0, β0 = 0.01, 0.01 # Inverse Gamma

    n = length(t)
    μ = -1e5
    σ = 1e5
    out = [(μ, σ)]
    for iter in 1:iterations
        # conjugate μ
        p0 = Gaussian{(:F,:Γ)}(mw, Λw)
        p = p0
        for i in n-1:-1:1
            Δ = t[i+1] - t[i]
            r = Mitosis.Regression([μ], (ConstantMap(Δ),), identity)
            k = kernel(Gaussian; μ=r, Σ=ConstantMap(σ^2*Δ))
            con = Mitosis.conjugate(k, x[i])
            _, pi = backward(BF(), con, [x[i+1]])
            _, p = fuse(p, pi)
        end

        # conjugate σ2
        α = α0
        β = β0
        μ = rand(convert(Gaussian{(:μ,:Σ)}, p))[]
        for i in n-1:-1:1
            Δ = t[i+1] - t[i]
            r = Mitosis.Regression([μ], (ConstantMap(Δ),), identity)
            s = Mitosis.Regression([σ^2], (ConstantMap(Δ),), zero)
            k = kernel(Gaussian; μ=r, Σ=s)
            α = α + 1/2
            β = β + (x[i+1] - mean(k(x[i])))^2/(2Δ)
        end
        σ = sqrt(rand(InverseGamma(α, β)))
        push!(out, (μ, σ))
    end
    out
end

chain = gibbs(t, x, 1000)
μs = first.(chain)
σs = last.(chain)

println("μ")
describe(μs)
println("σ")
describe(σs)

using Makie
# Chain mixing

pl1 = hbox(lines(getindex.(chain, 1)),
   lines(getindex.(chain, 2)))
# Make a small prediction

J = 2439809 # incumbent
C = 2430275 # challenger
N = C + J
T = (N - 36000)/N # remaining percentage to count
X = J - C # vote difference

s = range(T, 1.00-eps(), length=100)
μu = quantile(μs, 0.90)
μl = quantile(μs, 0.10)
μ = median(μs)
σu = quantile(σs, 0.9)
m = X .+ (s .- T)*μ
u = X .+ (s .- T)*μu .+ 2σu*sqrt.(s .- T)
l = X .+ (s .- T)*μl .- 2σu*sqrt.(s .- T)
Scene(resolution = (600, 600), textsize=20)
scatter!([T], [X])
lines!(s, u)
lines!(s, m)
lines!(s, l)
pl2 = band!(s, l, u)


pl2
