logpdf0(x, P) = logdensity(Gaussian{(:Σ,)}(P), x)


function right′(::BF, k::Union{AffineGaussianKernel,LinearGaussianKernel}, q::Gaussian{(:μ,:Σ)})
    ν, Σ = q.μ, q.Σ
    B, β, Q = params(k)
    B⁻¹ = inv(B)
    νp = B⁻¹*(ν - β)
    Σp = B⁻¹*(Σ + Q)*B⁻¹'
    q, Gaussian{(:μ,:Σ)}(νp, Σp)
end

function right′(::BFFG, k::Union{AffineGaussianKernel,LinearGaussianKernel}, q::WGaussian{(:F,:Γ,:c)}; unfused=false)
    @unpack F, Γ, c = q
    # Theorem 7.1 [Automatic BFFG]
    B, β, Q = params(k)
    Σ = inv(Γ) # requires invertibility of Γ
    K = B'/(Σ + Q)
    ν̃ = Σ*F - β
    Fp = K*ν̃
    Γp = K*B
    # Corollary 7.2 [Automatic BFFG]
    if !unfused
        cp = c - logdet(B)
    else
        cp = c + logpdf0(ν̃, Σ + Q)
    end
    message = q
    message, WGaussian{(:F,:Γ,:c)}(Fp, Γp, cp)
end

function right′(::BFFG, k::Union{AffineGaussianKernel,LinearGaussianKernel}, y; unfused=false)
    # Theorem 7.1 [Automatic BFFG]
    B, β, Q = params(k)
    K = B'/Q
    Fp = K*(y - β)
    Γp = K*B
    # Corollary 7.2 [Automatic BFFG]
    if !unfused
        cp = -logdet(B)
    else
        cp = +logpdf0(y - β, Q)
    end
    message = Leaf(y)
    message, WGaussian{(:F,:Γ,:c)}(Fp, Γp, cp)
end

function left′(::BFFG, k::Union{AffineGaussianKernel,LinearGaussianKernel}, m::WGaussian{(:F,:Γ,:c)})
    @unpack F, Γ, c = m
    B, β, Q = params(k)

    Q⁻ = inv(Q)
    Qᵒ = inv(Q⁻ + Γ)
    #μᵒ = Qᵒ*(Q⁻*(B*x + β) + F )
    Bᵒ = Qᵒ*Q⁻*B
    βᵒ = Qᵒ*(Q⁻*β + F)

    kernel(WGaussian; μ=AffineMap(Bᵒ, βᵒ), Σ=ConstantMap(Qᵒ), c=ConstantMap(0.0))
end

function left′(::BFFG, k::GaussKernel, k̃::Union{AffineGaussianKernel,LinearGaussianKernel}, m::WGaussian{(:F,:Γ,:c)}, x)
    @unpack F, Γ, c = m
    μ, Q = k.ops
    μ̃, Q̃ = k̃.ops

    # Proposition 7.3.
    Q⁻ = inv(Q(x))
    Qᵒ = inv(Q⁻ + Γ)
    μᵒ = Qᵒ*(Q⁻*(μ(x)) + F)

    Q̃⁻ = inv(Q̃(x))
    Q̃ᵒ = inv(Q̃⁻ + Γ)
    μ̃ᵒ = Q̃ᵒ*(Q̃⁻*(μ̃(x)) + F)

    c = logpdf0(μ(x), Q(x)) - logpdf0(μ̃(x), Q̃(x))
    c += logpdf0(μ̃ᵒ, Q̃ᵒ) - logpdf0(μᵒ, Qᵒ)

    WGaussian{(:μ,:Σ,:c)}(μᵒ, Qᵒ, c)
end


function right′(::BFFG, ::Copy, args::WGaussian{(:μ,:Σ,:c)}...; unfused=true)
    F, H, c = params(convert(WGaussian{(:F,:Γ,:c)}, args[1]))
    unfused || (c += logdensity(Gaussian{(:F,:Γ)}(F, H), 0F))
    for b in args[2:end]
        F2, H2, c2 = params(convert(WGaussian{(:F,:Γ,:c)}, b))
        F += F2
        H += H2
        c += c2
        unfused || (c += logdensity(Gaussian{(:F,:Γ)}(F2, H2), 0F2))
    end
    Δ = -logdensity(Gaussian{(:F,:Γ)}(F, H), 0F)
    m = ()
    m, convert(WGaussian{(:μ,:Σ,:c)}, WGaussian{(:F,:Γ,:c)}(F, H, Δ + c))
end

function right′(::BFFG, ::Copy, a::WGaussian{(:F,:Γ,:c)}, args...; unfused=true)
    F, H, c = params(a)
    unfused || (c += logdensity(Gaussian{(:F,:Γ)}(F, H), 0F))
    for b in args
        F2, H2, c2 = params(b::WGaussian{(:F,:Γ,:c)})
        F += F2
        H += H2
        c += c2
        unfused || (c += logdensity(Gaussian{(:F,:Γ)}(F2, H2), 0F2))
    end
    Δ = -logdensity(Gaussian{(:F,:Γ)}(F, H), 0F)
    m = ()
    m, WGaussian{(:F,:Γ,:c)}(F, H, Δ + c)
end

function left′(::BFFG, k::Union{AffineGaussianKernel,LinearGaussianKernel}, y::Leaf, x::Weighted)
    Dirac(weighted(y[], x.ll))

end
function left′(::BFFG, ::Copy{2}, _, x::Weighted)
    MeasureTheory.Dirac((x, weighted(x[])))
end
