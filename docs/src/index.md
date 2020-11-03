```@meta
CurrentModule = Mitosis
```

```@meta
DocTestSetup = quote
    using Mitosis
    using Random, Test, LinearAlgebra, Statistics
end
```

# Mitosis

Incorporate discrete and continuous time Markov processes as building blocks into probabilistic graphical models.

## Based on MeasureTheory.jl


Mitosis defines its probability distributions, densities in terms of
MeasureTheory.jl.

```jldoctest kernel
m = [1.0, 0.5]
K = Matrix(1.0I, 2, 2)
p = Gaussian(μ=m, Σ=K)

mean(p) == m

# output
true
```

## Key concepts

### Kernels or distribution valued maps

The core concept of Mitosis is the Markov [`kernel`](@ref).

A kernel `κ = kernel(Gaussian, μ=f, Σ=g)` returns a callable which returns a measure
with parameters determined by functions `f`, `g`...


```jldoctest kernel
f(x) = x*m
g(_) = K
k = kernel(Gaussian; μ=f, Σ=g)
mean(k(3.0)) == 3*m && cov(k(3.0)) == K

# output
true
```


### Linear and affine Gaussian kernel

Gaussian `kernel` become especially powerful if combined with linear and affine mean functions, [`AffineMap`](@ref), [`LinearMap`](@ref), [`ConstantMap`](@ref):

```jldoctest affine
B = [0.8 0.5; -0.1 0.8]
β = [0.1, 0.2]
Q = [0.2 0.0; 0.0 1.0]

x = [0.112, -1.22]
b = AffineMap(B, β)

b(x) == B*x + β

# output

true
```

Kernels with affine `mean` and constant covariance propagate Gaussian uncertainty:

```jldoctest affine

k = kernel(Gaussian, μ = AffineMap(B, β), Σ=ConstantMap(Q))

m = [1.0, 0.5]
K = Matrix(1.0I, 2, 2)
p = Gaussian(μ=m, Σ=K)

k(p) isa Gaussian

# output

true
```

## Example: Bayesian regression with `BF()`

Left and right functions with signature
`
```
message, marginal = right′(BF(), kernel, argument)
```

```
marginal = left′(BF(), kernel, message)(argument)
```

define a generic interface to a 2-pass backward filtering, forward smoothing algorithm.

`BF()` specifies the exact (conjugate) linear-Gaussian backward filter, forward smoothing version
without importance weights.

As illustration, a Bayesian regression example:

### Data

Small data set.

```jldoctest regression
x = [18.25 19.75 16.5 18.25 19.50 16.25 17.25 19.00 16.25 17.50][:]
y = [36 42 33 39 43 34 37 41 27 30][:]
n = length(x)

# output
10
```

### Model

Conditional on ``\beta``, a regression model:

``Y \mid \beta \sim N(X\beta, \sigma^2)``
where `X` is the design matrix.

Thus we can express this as linear Gaussian kernel:

```jldoctest regression
X = [x ones(n)] # Design matrix

σ2 = 8.0 # noise level
Σ = Diagonal(σ2*ones(n)) # noise covariance

model = kernel(Gaussian; μ=LinearMap(X), Σ=ConstantMap(Σ))

nothing
# output

```

### Prior

The conjugate prior is Gaussian,

``\beta \sim N(\mu_0, \sigma^2 V_0).``

We write it as kernel (without arguments) as well:

```jldoctest regression
μ0 = zeros(2)
V0 = 10*I(2)
Σ0 = σ2*V0 # prior

prior = kernel(Gaussian; μ=ConstantMap(μ0), Σ=ConstantMap(Σ0))

mean(prior()) == μ0

# output
true
```

Summarizing, our model says that,
```
β = rand(prior())
y = rand(model(β))
```
Think of this as the composition of kernels.

### Backward pass

The backward pass takes observations ``y`` into account and propagates uncertainty backward through the model.

```jldoctest regression
m2, p2 = right′(BF(), model, y)
m1, p1 = right′(BF(), prior, p2)

nothing
# output

```
### Forward pass

This `BF()` forward pass computes marginal distributions of latents.

```jldoctest regression
posterior = left′(BF(), prior, m1)()
# observations = left′(BF(), model, m2)(posterior)

mean(posterior), cov(posterior)

# output
([2.4874650784715016, -8.120051139323095], [0.1700435105146037 -3.00522441850067; -3.00522441850067 53.904213732907884])
```

## References

* Frank van der Meulen, Moritz Schauer (2020): Automatic Backward Filtering Forward Guiding for Markov processes and graphical models. [[arXiv:2010.03509]](https://arxiv.org/abs/2010.03509).
