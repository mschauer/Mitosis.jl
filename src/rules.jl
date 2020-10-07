function right′(::BFFG, k::GaussKernel, a)
	(c, F, H) = params(a)
	y, B, β = approx(k.ops[1], F, H)   # linearise mean function
	Q = k.ops[2](y)	            # constant approximation
	m = B, β, Q, Gaussian(F, H)	          # message
#         ...
#	c, F, H =  ...		     # compute new htilde
	return m, WeightedGaussian(c, F, H)
end
function left′(::BFFG, k::GaussKernel, (l, x), z, m)
	B, β, Q, a = m	           # unpack message
	F, H = params(a)
#	...			 	# compute parameters for guided proposal
#	c = ...
#	μᵒ =  ...
#	Qᵒ = ...
	return l + c, x + μᵒ + chol(Qᵒ)*z
end

function right′(::BFFG, ::Copy, a, b)
	(c1, F1, H1) = params(a)
	(c2, F2, H2) = params(b)
	Q = inv(H1)
	x = Q*F1
	Σ = inv(H2)
	ν = Σ*F2 # Kalman gain
	S = Q + Σ # innovation covariance
	K = Q/S
	x = x + K*(ν - x) P = (I - K)*Q

	Δ = logpdf0(x, P) - logpdf0(x, Q) - logpdf0(v, Σ) H = inv(P)
	m = ()
	m, WeightedGaussian(Δ + c1 + c2, H*x, H)
end
