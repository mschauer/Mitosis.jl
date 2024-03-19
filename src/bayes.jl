
function conditional(p::Gaussian{(:μ, :Σ)}, A, B)
    Z = p.Σ[A,B]*inv(p.Σ[B,B])
    β = p.μ[A] - Z*p.μ[B]
    Σ = p.Σ[A,A] - Z*p.Σ[B,A]
    kernel(Gaussian; μ=AffineMap(Z, β), Σ=ConstantMap(Σ))
end

function marginal(p::Gaussian{(:μ, :Σ)}, A)
    Gaussian{(:μ, :Σ)}(p.μ[A], p.Σ[A,A])
end

function likelihood(k::AffineGaussianKernel, obs)
    q = backward(BFFG(), k, obs; unfused=true)[2].y
    F, Γ, c =  params(q)
    c0 = Mitosis.logdensity0(Gaussian{(:F,:Γ)}(F, Γ))
    WGaussian{(:F,:Γ, :c)}(F, Γ, c - c0)
end