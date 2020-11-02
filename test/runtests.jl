using Mitosis
using Test
const TEST = true
include("gauss.jl")
include("statespace.jl")

@testset "Example" begin
    include("../example/example.jl")
end
