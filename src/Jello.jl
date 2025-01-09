module Jello
include("main.jl")

greet() = print("Hello World!")
export Blob, ConvBlob, InterpBlob, FourierBlob, AreaChangeOptimiser
export update_loss!, holesize
end # module Jello
