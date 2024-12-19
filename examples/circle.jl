# training a model to match a circular pattern

# include("../src/main.jl")
using Jello
using Random, CairoMakie, Flux

Random.seed!(1)
n = 100
lvoid = 10
lsolid = 10
solid_frac = 0.5
m = Blob(n, n; solid_frac, lvoid, lsolid, symmetries=[])
# m = Blob(n, n; solid_frac, lvoid, lsolid, symmetries=[1,2], periodic=true)

# generate a sample
a = m()
display(heatmap(a))
@show extrema(a)
# error("stop here")

opt = AreaChangeOptimiser(m; maxchange=0.2)
opt_state = Flux.setup(opt, m)
circ = [norm([x, y] - [n, n] / 2) < n / 4 for x = 1:n, y = 1:n]
for i = 1:40
    global l, (dldm,) = Flux.withgradient(m) do m
        Flux.mae(circ, m())
    end
    println("($i)")
    println("loss: $l")
    println()

    update_loss!(opt, l)
    Flux.update!(opt_state, m, dldm)
    heatmap(m()) |> display
end

