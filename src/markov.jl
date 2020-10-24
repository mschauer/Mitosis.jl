import MeasureTheory.Kernel, MeasureTheory.kernel

const GaussKernel = MeasureTheory.Kernel{<:Gaussian}
#const Copy = MeasureTheory.Kernel{<:Dirac}

struct AffineMap{S,T}
    B::S
    β::T
end
(a::AffineMap)(x) = a.B*x + a.β
(a::AffineMap)(p::Gaussian) = Gaussian(μ = a.B*mean(p) + a.β, Σ = a.B*cov(p)*a.B')


struct LinearMap{T}
    B::T
end
(a::LinearMap)(x) = a.B*x
(a::LinearMap)(p::Gaussian) = Gaussian(μ = a.B*mean(p), Σ = a.B*cov(p)*a.B')



struct ConstantMap{T}
    x::T
end
(a::ConstantMap)(x) = a.x

const LinearGaussianKernel =  Kernel{T,NamedTuple{(:μ, :Σ),Tuple{A, C}}} where {T<:Gaussian, A<:LinearMap, C<:ConstantMap}
params(k::LinearGaussianKernel) = k.ops.μ.B, Zero(), k.ops.Σ.x
const AffineGaussianKernel =  Kernel{T,NamedTuple{(:μ, :Σ),Tuple{A, C}}} where {T<:Gaussian, A<:AffineMap, C<:ConstantMap}
params(k::AffineGaussianKernel) = k.ops.μ.B, k.ops.μ.β, k.ops.Σ.x

"""
    correct(prior, obskernel, obs) = u, yres, S

Joseph form correction step of a Kalman filter with `prior` state
and `obs` the observation with observation kernel
`obskernel = kernel(Gaussian; μ=LinearMap(H), Σ=ConstantMap(R))`
`H` is the observation operator and `R` the observation covariance. Returns corrected/conditional
distribution `u`, the residual and the innovation covariance.
See https://en.wikipedia.org/wiki/Kalman_filter#Update.
"""
function correct(u::Gaussian{T}, k::LinearGaussianKernel, y) where {T}
    x, Ppred = meancov(u)
    H = k.ops.μ.B
    R = k.ops.Σ.x
    (x, P), yres, S = correct_joseph((x, Ppred), H, R, y)
    Gaussian{T}(μ=x, Σ=P), yres, S
end

function correct_joseph((x, Ppred), H, R, y)
    yres = y - H*x # innovation residual
    S = (H*Ppred*H' + R) # innovation covariance

    K = Ppred*H'/S # Kalman gain
    x = x + K*yres
    P = (I - K*H)*Ppred*(I - K*H)' + K*R*K' #  Joseph form
    (x, P), yres, S
end

function smooth(Gs::T, Gf, Gpred, Φ) where {T}
    xs, Ps = meancov(Gs)
    xf, Pf = meancov(Gf)
    xpred, Ppred = meancov(Gpred)

    J = Pf*Φ'/Ppred # C/(C+w)
    xs = xf + J*(xs - xpred)
    Ps = Pf + J*(Ps - Ppred)*J'

    (xs, Ps)
end
