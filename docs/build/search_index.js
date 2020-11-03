var documenterSearchIndex = {"docs":
[{"location":"library.html#Library","page":"Library","title":"Library","text":"","category":"section"},{"location":"library.html","page":"Library","title":"Library","text":"Modules = [Mitosis]","category":"page"},{"location":"library.html#Mitosis.AffineMap","page":"Library","title":"Mitosis.AffineMap","text":"AffineMap(B, β)\n\nRepresents a function f = AffineMap(B, β) such that f(x) == B*x + β.\n\n\n\n\n\n","category":"type"},{"location":"library.html#Mitosis.BF","page":"Library","title":"Mitosis.BF","text":"BF()\n\nBackward filter for linear Gaussian systems parametrized by mean and covariance of the backward filtered marginal distribution.\n\n\n\n\n\n","category":"type"},{"location":"library.html#Mitosis.BFFG","page":"Library","title":"Mitosis.BFFG","text":"BFFG()\n\nBackward filter forward guiding context for non-linear Gaussian systems with h parametrized by WGaussian{(:F,:Γ,:c)}` (see Theorem 7.1 [Automatic BFFG].)\n\n\n\n\n\n","category":"type"},{"location":"library.html#Mitosis.ConstantMap","page":"Library","title":"Mitosis.ConstantMap","text":"ConstantMap(β)\n\nRepresents a function f = ConstantMap(β) such that f(x) == β.\n\n\n\n\n\n","category":"type"},{"location":"library.html#Mitosis.Gaussian","page":"Library","title":"Mitosis.Gaussian","text":"Gaussian{(:μ,:Σ)}\nGaussian{(:F,:Γ)}\n\nMitosis provides the measure Gaussian based on MeasureTheory.jl, with a mean μ and covariance Σ parametrization, or parametrised by natural parameters F = Γ μ, Γ = Σ⁻¹.\n\nUsage:\n\nGaussian(μ=m, Σ=C)\np = Gaussian{(:μ,:Σ)}(m, C)\nGaussian(F=C\\m, Γ=inv(C))\n\nconvert(Gaussian{(:F,:Γ)}, p)\n\nrand(rng, p)\n\n\n\n\n\n","category":"type"},{"location":"library.html#Mitosis.LinearMap","page":"Library","title":"Mitosis.LinearMap","text":"LinearMap(B)\n\nRepresents a function f = LinearMap(B) such that f(x) == B*x.\n\n\n\n\n\n","category":"type"},{"location":"library.html#MeasureTheory.kernel","page":"Library","title":"MeasureTheory.kernel","text":"kernel(f, M)\nkernel((f1, f2, ...), M)\n\nA kernel κ = kernel(f, M) returns a wrapper around a function f giving the parameters for a measure of type M, such that κ(x) = M(f(x)...) respective κ(x) = M(f1(x), f2(x), ...).\n\nIf the argument is a named tuple (;a=f1, b=f1), κ(x) is defined as M(;a=f(x),b=g(x)).\n\nReference\n\nhttps://en.wikipedia.org/wiki/Markov_kernel\n\n\n\n\n\n","category":"function"},{"location":"library.html#Mitosis.conditional-Tuple{Gaussian{(:μ, :Σ),T} where T,Any,Any,Any}","page":"Library","title":"Mitosis.conditional","text":"conditional(p::Gaussian, A, B, xB)\n\nConditional distribution of X[i for i in A] given X[i for i in B] == xB if X  P.\n\n\n\n\n\n","category":"method"},{"location":"library.html#Mitosis.correct-Union{Tuple{T}, Tuple{Gaussian{T,T1} where T1,Kernel{T,NamedTuple{(:μ, :Σ),Tuple{A,C}}} where C<:ConstantMap where A<:LinearMap where T<:Gaussian,Any}} where T","page":"Library","title":"Mitosis.correct","text":"correct(prior, obskernel, obs) = u, yres, S\n\nJoseph form correction step of a Kalman filter with prior state and obs the observation with observation kernel obskernel = kernel(Gaussian; μ=LinearMap(H), Σ=ConstantMap(R)) H is the observation operator and R the observation covariance. Returns corrected/conditional distribution u, the residual and the innovation covariance. See https://en.wikipedia.org/wiki/Kalman_filter#Update.\n\n\n\n\n\n","category":"method"},{"location":"idx.html","page":"Index","title":"Index","text":"CurrentModule = Mitosis","category":"page"},{"location":"idx.html#Index","page":"Index","title":"Index","text":"","category":"section"},{"location":"idx.html","page":"Index","title":"Index","text":"","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"CurrentModule = Mitosis","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"DocTestSetup = quote\n    using Mitosis\n    using Random, Test, LinearAlgebra, Statistics\nend","category":"page"},{"location":"index.html#Mitosis","page":"Home","title":"Mitosis","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"Incorporate discrete and continuous time Markov processes as building blocks into probabilistic graphical models.","category":"page"},{"location":"index.html#Based-on-MeasureTheory.jl","page":"Home","title":"Based on MeasureTheory.jl","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"Mitosis defines its probability distributions, densities in terms of MeasureTheory.jl.","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"m = [1.0, 0.5]\nK = Matrix(1.0I, 2, 2)\np = Gaussian(μ=m, Σ=K)\n\nmean(p) == m\n\n# output\ntrue","category":"page"},{"location":"index.html#Key-concepts","page":"Home","title":"Key concepts","text":"","category":"section"},{"location":"index.html#Kernels-or-distribution-valued-maps","page":"Home","title":"Kernels or distribution valued maps","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"The core concept of Mitosis is the Markov kernel.","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"A kernel κ = kernel(Gaussian, μ=f, Σ=g) returns a callable which returns a measure with parameters determined by functions f, g...","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"f(x) = x*m\ng(_) = K\nk = kernel(Gaussian; μ=f, Σ=g)\nmean(k(3.0)) == 3*m && cov(k(3.0)) == K\n\n# output\ntrue","category":"page"},{"location":"index.html#Linear-and-affine-Gaussian-kernel","page":"Home","title":"Linear and affine Gaussian kernel","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"Gaussian kernel become especially powerful if combined with linear and affine mean functions, AffineMap, LinearMap, ConstantMap:","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"B = [0.8 0.5; -0.1 0.8]\nβ = [0.1, 0.2]\nQ = [0.2 0.0; 0.0 1.0]\n\nx = [0.112, -1.22]\nb = AffineMap(B, β)\n\nb(x) == B*x + β\n\n# output\n\ntrue","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"Kernels with affine mean and constant covariance propagate Gaussian uncertainty:","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"\nk = kernel(Gaussian, μ = AffineMap(B, β), Σ=ConstantMap(Q))\n\nm = [1.0, 0.5]\nK = Matrix(1.0I, 2, 2)\np = Gaussian(μ=m, Σ=K)\n\nk(p) isa Gaussian\n\n# output\n\ntrue","category":"page"},{"location":"index.html#Backward-and-forward-passes","page":"Home","title":"Backward and forward passes","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"Backward and forward functions with signature `","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"message, marginal = backward(BF(), kernel, argument)","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"marginal = forward(BF(), kernel, message)(argument)","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"define a generic interface to a 2-pass backward filtering, forward smoothing algorithm. For each transition, the backward pass produces a message for the forward pass.","category":"page"},{"location":"index.html#Example:-Bayesian-regression-with-BF()","page":"Home","title":"Example: Bayesian regression with BF()","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"BF() specifies the exact (conjugate) linear-Gaussian backward filter, forward smoothing version without importance weights.","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"As illustration, a Bayesian regression example:","category":"page"},{"location":"index.html#Data","page":"Home","title":"Data","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"Small data set.","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"x = [18.25 19.75 16.5 18.25 19.50 16.25 17.25 19.00 16.25 17.50][:]\ny = [36 42 33 39 43 34 37 41 27 30][:]\nn = length(x)\n\n# output\n10","category":"page"},{"location":"index.html#Prior","page":"Home","title":"Prior","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"The conjugate prior is Gaussian,","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"beta sim N(mu_0 sigma^2 V_0)","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"We write it as kernel (without arguments) as well:","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"σ2 = 8.0 # noise level\n\nμ0 = zeros(2)\nV0 = 10*I(2)\nΣ0 = σ2*V0 # prior\n\nprior = kernel(Gaussian; μ=ConstantMap(μ0), Σ=ConstantMap(Σ0))\n\nmean(prior()) == μ0\n\n# output\ntrue","category":"page"},{"location":"index.html#Model","page":"Home","title":"Model","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"Conditional on beta, a regression model:","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"Y mid beta sim N(Xbeta sigma^2) where X is the design matrix.","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"Thus we can express this as linear Gaussian kernel:","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"X = [x ones(n)] # Design matrix\n\n\nΣ = Diagonal(σ2*ones(n)) # noise covariance\n\nmodel = kernel(Gaussian; μ=LinearMap(X), Σ=ConstantMap(Σ))\n\nnothing\n# output\n","category":"page"},{"location":"index.html#Forward-model","page":"Home","title":"Forward model","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"Summarizing, our model says that,","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"β = rand(prior())\ny = rand(model(β))","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"Think of this as the composition of kernels.","category":"page"},{"location":"index.html#Backward-pass","page":"Home","title":"Backward pass","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"The backward pass takes observations y into account and propagates uncertainty backward through the model.","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"m2, p2 = backward(BF(), model, y)\nm1, p1 = backward(BF(), prior, p2)\n\nnothing\n# output\n","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"At each step it produces a filtered distribution p1, p2 and a message m1, m2 for the forward pass.","category":"page"},{"location":"index.html#Forward-pass","page":"Home","title":"Forward pass","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"This BF() forward pass computes marginal distributions of latents.","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"posterior = forward(BF(), prior, m1)()\n# observations = forward(BF(), model, m2)(posterior)\n\nmean(posterior), cov(posterior)\n\n# output\n([2.4874650784715016, -8.120051139323095], [0.1700435105146037 -3.00522441850067; -3.00522441850067 53.904213732907884])","category":"page"},{"location":"index.html#References","page":"Home","title":"References","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"Frank van der Meulen, Moritz Schauer (2020): Automatic Backward Filtering Forward Guiding for Markov processes and graphical models. [arXiv:2010.03509].","category":"page"}]
}
