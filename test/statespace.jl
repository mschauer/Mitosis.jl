#using Revise
using Mitosis


using Mitosis: kernel, Kernel, Gaussian, AffineMap, GaussKernel
using Random, Test, LinearAlgebra, Statistics

Random.seed!(1)
K = 10000 # repetitions

# State space system
#
# x[0] ∼ N(x0, P0)
# x[k] = Φx[k−1] + w[k],    w[k] ∼ N(0, Q)
# y[k] = Hx[k] + v[k],    v[k] ∼ N(0, R)

x0 = [1., 0.]
P0 = Matrix(1.0I, 2, 2)

Φ = [0.8 0.5; -0.1 0.8]
β = [0.1, 0.2]
Q = [0.2 0.0; 0.0 1.0]

yshadow = [0.0]
H = [1.0 0.0]
R = Matrix(1.0I, 1, 1)
INoise = Gaussian(μ=zero(x0), Σ=Q)
Noise = Gaussian(μ=zero(yshadow),Σ=R)

@test AffineMap(Φ, β)(x0) == Φ*x0 + β

transition = kernel(Gaussian; μ=AffineMap(Φ, β), Σ=(_)->Q)
observation = kernel(Gaussian; μ=AffineMap(H, yshadow), Σ=(_)->R)

@test 10/sqrt(K) > norm(Φ*x0 + β - mean(rand(transition(x0)) for x in 1:K))

@test transition.ops isa NamedTuple{(:μ, :Σ)}

y0 = rand(observation(x0))

x1 = rand(transition(x0))
y1 = rand(observation(x1))

x2 = rand(transition(x1))
y2 = rand(observation(x2))




# Check that this is all correct:

# Write down joint distribution of x's and y's
# Define mean and covariance of the flattened vector of states and observations [x0 x1 x2 y0 y1 y2]
μ = [1.0, 0.0, 0.8, -0.1, 0.59, -0.16, 1.0, 0.8, 0.59]
Σ = [1.0   0.0   0.8    -0.1    0.59    -0.16     1.0   0.8    0.59
    0.0   1.0   0.5     0.8    0.8      0.59     0.0   0.5    0.8
    0.8   0.5   1.09    0.32   1.032    0.147    0.8   1.09   1.032
    -0.1   0.8   0.32    1.65   1.081    1.288   -0.1   0.32   1.081
    0.59  0.8   1.032   1.081  1.5661   0.7616   0.59  1.032  1.5661
    -0.16  0.59  0.147   1.288  0.7616   2.0157  -0.16  0.147  0.7616
    1.0   0.0   0.8    -0.1    0.59    -0.16     2.0   0.8    0.59
    0.8   0.5   1.09    0.32   1.032    0.147    0.8   2.09   1.032
    0.59  0.8   1.032   1.081  1.5661   0.7616   0.59  1.032  2.5661]

# Compute the conditional distribution of vector `x0` given data.
xtest = Mitosis.conditional(Gaussian(;μ=μ, Σ=Σ), 1:2, 7:9, vcat(y0, y1, y2))
