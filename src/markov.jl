import MeasureTheory.Kernel, MeasureTheory.kernel

const GaussKernel = MeasureTheory.Kernel{<:Gaussian}
const Copy = MeasureTheory.Kernel{<:Dirac}

struct AffineMap{S,T}
    B::S
    β::T
end
(a::AffineMap)(x) = a.B*x + a.β
