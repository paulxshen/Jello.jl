# training a model to match a circular pattern

# include("../src/main.jl")
using Jello
using Random, CairoMakie, Flux, LinearAlgebra
Random.seed!(1)

n = 100
lvoid = 10
lsolid = 10
solid_frac = 0.5
symmetries = [1, 2, :diagonal]

# generate a sample
m = Blob(n, n; solid_frac, lvoid, lsolid, symmetries)
display(heatmap(m()))
# m = Blob(n, n;  lvoid, lsolid, symmetries=[1,2], periodic=true)
# display(heatmap(m()))

error("stop here")

opt = AreaChangeOptimiser(m)
opt_state = Flux.setup(opt, m)
circ = [norm([x, y] - [n, n] / 2) < n / 4 for x = 1:n, y = 1:n]
for i = 1:20
    global l, (dldm,) = Flux.withgradient(m) do m
        Flux.mae(circ, m())
    end
    println("($i)")
    println("loss: $l")

    opt.minchange = max(0.001, 0.2l^2)
    update_loss!(opt, l)
    Flux.update!(opt_state, m, dldm)
    heatmap(m()) |> display
    println()
end

