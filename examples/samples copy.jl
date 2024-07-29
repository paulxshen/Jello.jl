using Random, LinearAlgebra
# using Jello
include("../src/main.jl")

fig2d = Figure()
l = 100

Random.seed!(1)
contrast = 1
rmin = nothing
m = Blob(l, l; contrast)
