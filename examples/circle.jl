# training a model to match a circular pattern

using Random, CairoMakie, Flux
# using Jello
include("../src/main.jl")

Random.seed!(1)
n = 100
lvoid = 10
lsolid = 10
solid_frac = 0.5
m = Blob(n, n; solid_frac, lvoid, lsolid, symmetries=[])
# m = Blob(n, n; solid_frac, lvoid, lsolid, symmetries=[1, 2])

# generate a sample
a = m()
display(heatmap(a))
# error("stop here")

opt = AreaChangeOptimiser(m; maxchange=0.2)
opt_state = Flux.setup(opt, m)
for i = 1:20
    global l, (dldm,) = Flux.withgradient(m) do m
        circ = [norm([x, y] - [n, n] / 2) < n / 4 for x = 1:n, y = 1:n]
        Flux.mae(circ, m())
    end
    println("($i)")
    println("loss: $l")
    update_loss!(opt, l)
    Flux.update!(opt_state, m, dldm)
    println()
    heatmap(m()) |> display
end

