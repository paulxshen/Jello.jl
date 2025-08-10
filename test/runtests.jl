# training a model to match a circular pattern
# ENV["JULIA_DEBUG"] = "Main"
# include("../src/main.jl")
using Jello
using Random, CairoMakie, Flux, LinearAlgebra
Random.seed!(1)

n = 100
lmin = n / 10
# init = 0.5
init = 0.5
# init = zeros(n, n)
# init[:, 40:60] .= 1
# init[1:40, 1:40] .= 2
symmetries = [:x, :inversion]
# symmetries = []
contrast = 0.9

# generate a sample
m = Blob(n, n; init, lmin, symmetries, contrast)
display(heatmap(m()))
# m = Blob(n, n;  lmin, lsolid, symmetries=[1,2], periodic=true)
# display(heatmap(m()))

error("stop here")

opt = opt_state = nothing

invert(x, b) = b ? 1 - x : x
c = [n, n] / 2 + 0.5
R = [0.1, 0.2, 0.3] * n
circ = map(CartesianIndices((n, n + 1))) do I
    I = Tuple(I)
    inverted = false
    for r = R
        v = norm(I - c) - r
        v < 0 && return invert(1, inverted)
        v < 1 && return invert(1 - v, inverted)
        inverted = !inverted
    end
    invert(1, inverted)
end
heatmap(circ) |> display
error("stop here")

for i = 1:20
    global l, (dldm,) = Flux.withgradient(m) do m
        Flux.mae(circ, m())
    end
    println("($i)")
    println("loss: $l")

    global opt, opt_state
    if isnothing(opt)
        η = η_from_area_change(m, dldm, 0.1)
        opt = Descent(η)
        opt_state = Flux.setup(opt, m)
    end
    Flux.update!(opt_state, m, dldm)
    heatmap(m()) |> display
    println()
end
