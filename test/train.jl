# using Random, Jello, CairoMakie
using Random, CairoMakie
include("../src/main.jl")

Random.seed!(1)
l = 100
lvoid = 1
lsolid = 1
init = nothing
# init = 1
m = Blob(l, l; init, lvoid, lsolid)
# m = gpu(m)
sharpness = 0.99
a = m(sharpness)

fig = Figure()
grid = fig[1, 1]
ax, plt = heatmap(grid[1, 1], m())
Colorbar(grid[1, 2], plt)
display(fig)