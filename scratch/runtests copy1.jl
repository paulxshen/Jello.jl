using Test, Random, Flux, CairoMakie, LinearAlgebra
include("../src/Jello.jl")
Random.seed!(1)

fig = Figure()
l = 128
dims = (l, l)

for (i, lmin) = enumerate(round.(Int, [l / 8, l / 12]))
    m = Mask(dims, lmin)
    for (j, contrast) = enumerate([0.1, 0.3])
        axis = (; title="$l x $l\nlmin = $lmin\ncontrast = $contrast\n")
        heatmap(fig[i, j], m(contrast); axis)
    end
end
display(fig)
