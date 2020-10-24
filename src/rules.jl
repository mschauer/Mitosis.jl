logpdf0(x, P) = logdensity(Gaussian{(:Σ,)}(P), x)


function right′(::BF, k::LinearGaussianKernel, q::Gaussian{(:μ, :Σ)})
    ν, Σ = q.μ, q.Σ
    B, _, Q = params(k)
    B⁻¹ = inv(B)
    νp = B⁻¹*ν
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
    #μᵒ = Qᵒ*(Q⁻*(B*x + β) + F)
    Bᵒ = Qᵒ*Q⁻*B
    βᵒ = Qᵒ*(Q⁻*β + F)

    kernel(WGaussian; μ=AffineMap(Bᵒ, βᵒ), Σ=ConstantMap(Qᵒ), c=ConstantMap(0.0))
end

function right′(::BFFG, k::GaussKernel, a)
    (c, F, H) = wgaussian_params(a)
    y, B, β = approx(k.ops[1], F, H)   # linearise mean function
    Q = k.ops[2](y)                # constant approximation
    m = B, β, Q, Gaussian(F, H)              # message
#         ...
#    c, F, H =  ...             # compute new htilde
    return m, WeightedGaussian(c, F, H)
end
function left′(::BFFG, k::GaussKernel, (l, x), z, m)
    B, β, Q, a = m               # unpack message
    _, F, H = wgaussian_params(a)
#    ...                 # compute parameters for guided proposal
#    c = ...
#    μᵒ =  ...
#    Qᵒ = ...
    return nothing, (l + c, x + μᵒ + chol(Qᵒ)*z)
end



function right′(::BFFG, ::Copy, a::WGaussian{(:F, :Γ, :c)}, args...)
    F, H, c = params(a)
    for b in args
        F2, H2, c2 = params(b::WGaussian{(:F, :Γ, :c)})
        F += F2
        H += H2
        c += c2
    end
    Δ = -logdensity(Gaussian{(:F,:Γ)}(F, H), 0F)
    m = ()
    m, WGaussian{(:F,:Γ,:c)}(F, H, Δ + c)
end

function left′(::BFFG, ::Copy{2}, (l, x), _, m)
    (l, x), (0l, x)
end
