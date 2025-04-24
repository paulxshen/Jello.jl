# training a model to match a circular pattern
ENV["JULIA_DEBUG"] = "Main"
ENV["JULIA_PKG_PRECOMPILE_AUTO"] = 0
include("../src/main.jl")
# using Jello
using Random, CairoMakie, Flux, LinearAlgebra
Random.seed!(1)

n = 100
lmin = n / 10
init = 0.5
# init = zeros(n, n)
# init[:, 40:60] .= 1
# init[1:40, 1:40] .= 2
symmetries = [:x, :y, :diagonal]
# symmetries = ["x"]
symmetries = []

# generate a sample
m = Blob(n, n; init, lmin, symmetries)
display(heatmap(m()))
# m = Blob(n, n;  lmin, lsolid, symmetries=[1,2], periodic=true)
# display(heatmap(m()))

# error("stop here")

opt = AreaChangeOptimiser(m, 0.1)
opt_state = Flux.setup(opt, m)
c = [n, n] / 2 + 0.5
circ = map(CartesianIndices((n, n))) do I
    I = Tuple(I)
    v = norm(I - c) - n / 4
    v < 0 && return 1
    v > 1 && return 0
    1 - v
end
heatmap(circ) |> display

for i = 1:30
    global l, (dldm,) = Flux.withgradient(m) do m
        Flux.mae(circ, m())
    end
    println("($i)")
    println("loss: $l")

    push!(opt.losses, l)
    Flux.update!(opt_state, m, dldm)
    heatmap(m()) |> display
    println()
end
