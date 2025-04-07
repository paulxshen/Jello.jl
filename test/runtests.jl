# training a model to match a circular pattern
ENV["JULIA_DEBUG"] = "Main"

include("../src/main.jl")
# using Jello
using Random, CairoMakie, Flux, LinearAlgebra
Random.seed!(1)

n = 100
lmin = 5
init = 0.5
# init = zeros(n, n)
# init[:, 40:60] .= 1
# init[1:40, 1:40] .= 2
# symmetries = [:x, :diagonal]
# symmetries = ["x"]
symmetries = []

# generate a sample
m = Blob(n, n; init, lmin, symmetries)
display(heatmap(m()))
# m = Blob(n, n;  lmin, lsolid, symmetries=[1,2], periodic=true)
# display(heatmap(m()))

error("stop here")

opt = AreaChangeOptimiser(m, 0.05)
opt_state = Flux.setup(opt, m)
circ = [norm([x, y] - [n, n] / 2) < n / 4 for x = 1:n, y = 1:n]
for i = 1:20
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
