logpdf0(x, P) = logdensity(Gaussian{(:Σ,)}(P), x)

struct Message{T,S}
    q0::S
    q::T
end
message(q0, q) = Message(q0, q)
message() = nothing

function backward(::BF, k::Union{AffineGaussianKernel,LinearGaussianKernel}, q::Gaussian{(:μ,:Σ)})
    ν, Σ = q.μ, q.Σ
    B, β, Q = params(k)
    B⁻¹ = inv(B)
    νp = B⁻¹*(ν - β)
    Σp = B⁻¹*(Σ + Q)*B⁻¹'
    q0 = Gaussian{(:μ,:Σ)}(νp, Σp)
    message(q0, q), q0
end

function backward(::BF, k::ConstantGaussianKernel, q::Gaussian{(:F,:Γ)})
    message(nothing, q), nothing
end


function backward(::BF, k::Union{AffineGaussianKernel,LinearGaussianKernel}, q::Gaussian{(:F,:Γ)})
    @unpack F, Γ = q
    # Theorem 7.1 [Automatic BFFG]
    B, β, Q = params(k)
    Σ = inv(Γ) # requires invertibility of Γ
    K = B'*inv(Σ + Q)
    ν̃ = Σ*F - β
    Fp = K*ν̃
    Γp = K*B
    q0 = Gaussian{(:F,:Γ)}(Fp, Γp)
    message(q0, q), q0
end


function backward(::BF, k::Union{AffineGaussianKernel,LinearGaussianKernel}, y)
    # Theorem 7.1 [Automatic BFFG]
    B, β, Q = params(k)
    K = B'/Q
    Fp = K*(y - β)
    Γp = K*B
    q0 = Gaussian{(:F,:Γ)}(Fp, Γp)
    message(q0, Leaf(y)), q0
end

backward(method::BFFG, k::Union{AffineGaussianKernel,LinearGaussianKernel}, q::Leaf; kargs...) = backward(method, k, q[]; kargs...)
backward(method::BFFG, k, q::Leaf; kargs...) = backward(method, k, q[]; kargs...)


function backward(::BFFG, k::Union{AffineGaussianKernel,LinearGaussianKernel}, q::WGaussian{(:F,:Γ,:c)}; unfused=false)
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
        cp = c - logdensity0(Gaussian{(:F,:Γ)}(Fp, Γp)) + logpdf0(ν̃, Σ + Q)
    end
    q0 = WGaussian{(:F,:Γ,:c)}(Fp, Γp, cp)
    message(q0, q), q0
end

function backward(::BFFG, k::Union{AffineGaussianKernel,LinearGaussianKernel}, y; unfused=false)
    # Theorem 7.1 [Automatic BFFG]
    B, β, Q = params(k)
    K = B'/Q
    Fp = K*(y - β)
    Γp = K*B
    # Corollary 7.2 [Automatic BFFG]
    if !unfused
        cp = -logdet(B)
    else
        cp = logpdf0(y - β, Q)
    end
    q0 = WGaussian{(:F,:Γ,:c)}(Fp, Γp, cp)
    message(q0, Leaf(y)), Leaf(q0)
end


function forward(::BF, k::Union{AffineGaussianKernel,LinearGaussianKernel}, m::Message{<:Gaussian{(:F,:Γ)}})
    @unpack F, Γ = m.q
    B, β, Q = params(k)

    Q⁻ = inv(Q)
    Qᵒ = inv(Q⁻ + Γ)
    Bᵒ = Qᵒ*Q⁻*B
    βᵒ = Qᵒ*(Q⁻*β + F)

    kernel(Gaussian; μ=AffineMap(Bᵒ, βᵒ), Σ=ConstantMap(Qᵒ))
end
function forward(::BF, k::ConstantGaussianKernel, m::Message{<:Gaussian{(:F,:Γ)}})
    @unpack F, Γ = m.q
    β, Q = params(k)

    Q⁻ = inv(Q)
    Qᵒ = inv(Q⁻ + Γ)
    βᵒ = Qᵒ*(Q⁻*β + F)

    kernel(Gaussian; μ=ConstantMap(βᵒ), Σ=ConstantMap(Qᵒ))
end



function forward(::BFFG, k::Union{AffineGaussianKernel,LinearGaussianKernel}, m::Message{<:WGaussian{(:F,:Γ,:c)}})
    @unpack F, Γ, c = m.q
    B, β, Q = params(k)

    Q⁻ = inv(Q)
    Qᵒ = inv(Q⁻ + Γ)
    #μᵒ = Qᵒ*(Q⁻*(B*x + β) + F )
    Bᵒ = Qᵒ*Q⁻*B
    βᵒ = Qᵒ*(Q⁻*β + F)

    kernel(WGaussian; μ=AffineMap(Bᵒ, βᵒ), Σ=ConstantMap(Qᵒ), c=ConstantMap(0.0))
end

function forward(bffg::BFFG, k::Kernel, m::Message, x::Weighted)
    p = forward_(bffg, k, m, x[])
    weighted(p, x.ll)
end
forward(bffg::BFFG, k::Kernel, m::Message, x) = forward_(bffg, k, m, x)
function forward_(::BFFG, k::GaussKernel, m::Message{<:WGaussian{(:F,:Γ,:c)}}, x)
    @unpack F, Γ, c = m.q
    c1 = c
    μ, Q = k.ops


    # Proposition 7.3.
    Q⁻ = inv(Q(x))
    Qᵒ = inv(Q⁻ + Γ)
    μᵒ = Qᵒ*(Q⁻*(μ(x)) + F)

 #   Q̃⁻ = inv(Q̃(x))
 #   Q̃ᵒ = inv(Q̃⁻ + Γ)
 #   μ̃ᵒ = Q̃ᵒ*(Q̃⁻*(μ̃(x)) + F)

 #   c = logpdf0(μ(x), Q(x)) - logpdf0(μ̃(x), Q̃(x))
 #   c += logpdf0(μ̃ᵒ, Q̃ᵒ) - logpdf0(μᵒ, Qᵒ)
 #   == logpdf0(μ(x) - Γ\F, Q(x) + inv(Γ)) - logpdf0(μ̃(x)  - Γ\F, Q̃(x) + inv(Γ)) 
    
 #   logdensity(m.q0, x) - c1 == logpdf0(μ̃(x)  - Γ\F, Q̃(x) + inv(Γ)) 

    c = logpdf0(μ(x) - Γ\F, Q(x) + inv(Γ)) - logdensity(m.q0, x) + c1
    WGaussian{(:μ,:Σ,:c)}(μᵒ, Qᵒ, c)
end


function backward(::BFFG, ::Copy, args::Union{Leaf{<:WGaussian{(:μ,:Σ,:c)}},WGaussian{(:μ,:Σ,:c)}}...; unfused=true)
    unfused = false
    F, H, c = params(convert(WGaussian{(:F,:Γ,:c)}, args[1]))
    args[1] isa Leaf || (c += logdensity0(Gaussian{(:F,:Γ)}(F, H)))
    for b in args[2:end]
        F2, H2, c2 = params(convert(WGaussian{(:F,:Γ,:c)}, b))
        F += F2
        H += H2
        c += c2
        b isa Leaf|| (c += logdensity0(Gaussian{(:F,:Γ)}(F2, H2)))
    end
    Δ = -logdensity(Gaussian{(:F,:Γ)}(F, H), 0F)
    
    message(), convert(WGaussian{(:μ,:Σ,:c)}, WGaussian{(:F,:Γ,:c)}(F, H, Δ + c))
end


function backward(::Union{BFFG,BF}, ::Copy, a::Gaussian{(:F,:Γ)}, args...)
    F, H = params(a)
    for b in args
        F2, H2 = params(b::Gaussian{(:F,:Γ)})
        F += F2
        H += H2
    end
    message(), Gaussian{(:F,:Γ)}(F, H)
end

function backward(::BFFG, ::Copy, a::Union{Leaf{<:WGaussian{(:F,:Γ,:c)}}, WGaussian{(:F,:Γ,:c)}}, args...; unfused=true)
    unfused = false
    F, H, c = params(convert(WGaussian{(:F,:Γ,:c)}, a))
    a isa Leaf || (c += logdensity(Gaussian{(:F,:Γ)}(F, H), 0F))
    for b in args
        F2, H2, c2 = params(convert(WGaussian{(:F,:Γ,:c)}, b))
        F += F2
        H += H2
        c += c2
        b isa Leaf || (c += logdensity(Gaussian{(:F,:Γ)}(F2, H2), 0F2))
    end
    Δ = -logdensity(Gaussian{(:F,:Γ)}(F, H), 0F)
    message(), WGaussian{(:F,:Γ,:c)}(F, H, Δ + c)
end

function forward(::BFFG, k::Union{AffineGaussianKernel,LinearGaussianKernel}, m::Message{<:Leaf}, x::Weighted)
    y = m.q
    Dirac(weighted(y[], x.ll))
end
function forward(::BFFG, ::Copy{2}, _, x::Weighted)
    Dirac((x, weighted(x[])))
end
