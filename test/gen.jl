using Random, Jello, CairoMakie

Random.seed!(1)
l = 100
lvoid = 6
lsolid = 6
init = nothing
m = Blob(l, l; init, lvoid, lsolid)
# m = gpu(m)
sharpness = 0.99
a = m(sharpness)

fig = Figure()
grid = fig[1, 1]
ax, plt = heatmap(grid[1, 1], m())
Colorbar(grid[1, 2], plt)
display(fig)