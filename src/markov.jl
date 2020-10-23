import MeasureTheory.Kernel, MeasureTheory.kernel

const GaussKernel = MeasureTheory.Kernel{<:Gaussian}
const Copy = MeasureTheory.Kernel{<:Dirac}

struct AffineMap{S,T}
    B::S
    β::T
end
(a::AffineMap)(x) = a.B*x + a.β
(a::AffineMap)(p::Gaussian) = Gaussian(μ = a.B*mean(p) + a.β, Σ = a.B*cov(p)*a.B')


struct ConstantMap{T}
    x::T
end
(a::ConstantMap)(x) = a.x

"""
    correct(prior, kernel, obs) = u, yres, S

Joseph form correction step of a Kalman filter with `u = Gaussian(x, P)` state
and `v = Gaussian(y, R)` the observation with uncertainty `R`.
`H` is the observation operator. Returns corrected/conditional
distribution u, the residual and the innovation covariance.
See https://en.wikipedia.org/wiki/Kalman_filter#Update.
"""
function correct(u::T, k::Kernel{T2,NamedTuple{(:μ, :Σ),Tuple{A, C}}}, y) where {T, T2#=<:Gaussian=#, A<:AffineMap, C<:ConstantMap}
    x, Ppred = meancov(u)
    H, y0 = k.ops.μ.B, k.ops.μ.β
    R = k.ops.Σ.x

    yres = y - H*x # innovation residual
    @show H,Ppred, R
    S = (H*Ppred*H' + R) # innovation covariance

    K = Ppred*H'/S # Kalman gain
    x = x + K*yres
    P = (I - K*H)*Ppred*(I - K*H)' + K*R*K' #  Joseph form

    Gaussian(μ=x, Σ=P), yres, S
end
