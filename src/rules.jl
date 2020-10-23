function right′(::BFFG, k::LinearGaussianKernel, q::GaussianOrNdTuple{(:μ, :Σ)})
	ν, Σ = q.μ, q.Σ
	B, Q = params(k)
	B⁻¹ = inv(B)
    νp = B⁻¹*ν
    Σp = B⁻¹*(Σ + Q)*B⁻¹'
    #c = c + logdet(B*B')/2
    q, Gaussian{(:μ,:Σ)}(νp, Σp)
end

function right′(::BFFG, k::LinearGaussianKernel, q::WGaussianOrNdTuple{(:F, :Γ, :c)})
	F = big.(q.F)
	Γ = big.(q.Γ)
	c = big.(q.c)

	B, Q = params(k)
	Σ = inv(Γ) # requires invertibility of Σ
	K = B'/(Σ + Q)
	Fp = K*Σ*F
    Γp = K*B
    cp = c - logdet(B)
    q, WGaussian{(:F,:Γ,:c)}(Float64.(Fp), Float64.(Γp), Float64.(cp))
end


function right′(::BFFG, k::GaussKernel, a)
	(c, F, H) = wgaussian_params(a)
	y, B, β = approx(k.ops[1], F, H)   # linearise mean function
	Q = k.ops[2](y)	            # constant approximation
	m = B, β, Q, Gaussian(F, H)	          # message
#         ...
#	c, F, H =  ...		     # compute new htilde
	return m, WeightedGaussian(c, F, H)
end
function left′(::BFFG, k::GaussKernel, (l, x), z, m)
	B, β, Q, a = m	           # unpack message
	_, F, H = wgaussian_params(a)
#	...			 	# compute parameters for guided proposal
#	c = ...
#	μᵒ =  ...
#	Qᵒ = ...
	return nothing, (l + c, x + μᵒ + chol(Qᵒ)*z)
end

function right′(::BFFG, ::Copy, a, b)
	c1, Q, x = wgaussian_params(a)
	c2, Σ, ν = wgaussian_params(b)
	S = Q + Σ # innovation covariance
	K = Q/S # Kalman gain
	x = x + K*(ν - x)
	P = (I - K)*Q
	Δ = logpdf0(x, P) - logpdf0(x, Q) - logpdf0(v, Σ)
	H = inv(P)
	m = ()
	m, weightedgaussian(Δ + c1 + c2, H*x, H)
end
