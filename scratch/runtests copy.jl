using Test, Random, Flux, CairoMakie, LinearAlgebra
Random.seed!(1)
m = Mask((128, 128), 2)
heatmap(m())

n = 128
y = float.([norm(collect(v) - [64, 64]) < 20 for v = Iterators.product(axes(ones(n, n))...)])
loss(m, y) = 10Flux.mae(m(), y)
opt = Adam(0.1)
opt_state = Flux.setup(opt, m)
for i = 1:100
    Flux.train!(m, [(0, y)], opt_state) do m, x, y
        l = loss(m, y)
        println(l)
        l
    end
end
heatmap(m())
heatmap(y)
