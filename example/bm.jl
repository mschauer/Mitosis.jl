# Fit a Brownian motion with drift to and scale to
# Georgia election data:
# fraction of total count of votes vs difference in votes
using DelimitedFiles
datafile = download("https://alex.github.io/nyt-2020-election-scraper/battleground-state-changes.csv")
datatable, header = readdlm(datafile, ',', header=true)

ii = datatable[:,1] .== "Pennsylvania (EV: 20)"

ss = 2(datatable[ii,3] .== "Trump") .- 1
c = reverse(datatable[ii,5] + datatable[ii,6])
x = reverse(datatable[ii,7].*ss)

N = c[end]
# N = C + J
R = datatable[ii, 8][1]
T = (N - R)/N # remaining percentage to count
X = x[end] #vote difference

t = c ./ (R + N)

#=
(1, "state")
(2, "timestamp")
(3, "leading_candidate_name")
(4, "trailing_candidate_name")
(5, "leading_candidate_votes")
(6, "trailing_candidate_votes")
(7, "vote_differential")
(8, "votes_remaining")
(9, "new_votes")
(10, "new_votes_relevant")
(11, "new_votes_formatted")
(12, "leading_candidate_partition")
(13, "trailing_candidate_partition")
(14, "precincts_reporting")
(15, "precincts_total")
(16, "hurdle")
(17, "hurdle_change")
(18, "hurdle_mov_avg")
(19, "counties_partition")
(20, "total_votes_count")

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

=#


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
        #α = α0
        #β = β0
        α = 2n
        β = var(Binomial(N, (1-μ/N)/2))*α
        μ = rand(convert(Gaussian{(:μ,:Σ)}, p))[]
        for i in n-1:-1:1
            Δ = t[i+1] - t[i]
            r = Mitosis.Regression([μ], (ConstantMap(Δ),), identity)
            s = Mitosis.Regression([σ^2], (ConstantMap(Δ),), zero)
            k = kernel(Gaussian; μ=r, Σ=s)
            α = α + 1/2
            β = β + (x[i+1] - mean(k(x[i])))^2/(2Δ)
        end
        #σ = sqrt(rand(InverseGamma(α, β)))
        σ = std(Binomial(N, (1-μ/N)/2))
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

s = range(T, 1.00-eps(), length=100)
μu = quantile(μs, 0.80)
μl = quantile(μs, 0.20)
μ = median(μs)
σ = median(σs)

σu = quantile(σs, 0.8)
m = X .+ (s .- T)*μ
u = X .+ (s .- T)*μu .+ 2σu*sqrt.(s .- T)
l = X .+ (s .- T)*μl .- 2σu*sqrt.(s .- T)
Scene(resolution = (600, 600), textsize=20)
scatter!([T], [X])
#lines!(s, u)
lines!(s, m)
#lines!(s, l)
pl2 = band!(s, l, u)


v = [rand(kernel(Gaussian; μ=x->x+rand(μs)*(1-T), Σ=_->rand(σs)^2*(1-T))(X))
   for k in 1:1000]
mean(v .> 0)
mean(u[end] .> v .> l[end])
#scatter!(one.(v)+0.00001*randn(length(v)), v, markersize=0.1)
pl2


function bm(x, s, μ, σ)
    y = float(x)
    ys = [y]
    for i in 1:length(s)-1
        Δ = s[i+1] - s[i]
        r = Mitosis.Regression([μ], (ConstantMap(Δ),), identity)
        k = kernel(Gaussian; μ=r, Σ=ConstantMap(σ^2*Δ))
        y = rand(k(y))
        push!(ys, y)
    end
    ys
end

for i in 1:100
    lines!(s, bm(X, s, rand(μs), rand(σs)))
end
pl2


#
σ = std(Binomial(N, (1-μ/N)/2))
