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

## References

* Frank van der Meulen, Moritz Schauer (2020): Automatic Backward Filtering Forward Guiding for Markov processes and graphical models. [[arXiv:2010.03509]](https://arxiv.org/abs/2010.03509).
