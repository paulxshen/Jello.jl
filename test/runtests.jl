# training a model to match a circular pattern
# ENV["JULIA_DEBUG"] = "Main"
include("../src/main.jl")
# using Jello
using Random, LinearAlgebra, CairoMakie
Random.seed!(1)

n = 20
lmin = n / 10
init = 1
repdims=3
anchordims=[-1]

# generate a sample
m = Blob(n, n, n; lmin, init, repdims, anchordims)
a=m()
display(heatmap(a[:, :, 1]))
