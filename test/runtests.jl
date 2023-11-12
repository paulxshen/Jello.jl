using Test, Random, Flux, CairoMakie, LinearAlgebra
include("../src/Jello.jl")
Random.seed!(1)

l = 128
y = float.([norm([x, y] - [l, l] / 2) < l / 4 for x = 1:l, y = 1:l]) # circle
loss(m) = Flux.mae(m(0.1), y)

m = Mask((l, l), l / 16)
opt = Adam(0.1)
opt_state = Flux.setup(opt, m)

fig = Figure()
heatmap(fig[1, 1], m(0), axis=(; title="start of training"))
for i = 1:100
    Flux.train!(m, [[]], opt_state) do m, _
        l = loss(m)
        println(l)
        l
    end
end
heatmap(fig[1, 2], m(0), axis=(; title="end of training"))
display(fig)