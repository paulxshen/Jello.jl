module Jello
include("main.jl")

greet() = print("Hello World!")
export Blob, InterpBlob, FourierBlob, AreaChangeOptimiser
export update_loss!, jump
end # module Jello
