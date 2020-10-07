struct MarkovKernel{T,S}
    ops::S
    m::T
end
(k::MarkovKernel)(x) = k.m((op(x) for op in k.ops)...)

const GaussKernel = MarkovKernel{<:Gaussian}
const Copy = MarkovKernel{<:Dirac}
