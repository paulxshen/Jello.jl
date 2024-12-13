# training a model to match a circular pattern

# using Random, Jello, CairoMakie
using Random, CairoMakie, Flux
include("../src/main.jl")
include("../src/opt.jl")
# include("sgopt.jl")

Random.seed!(1)
n = 100
lvoid = 10
lsolid = 10
init = nothing # random
m = Blob(n, n; init, lvoid, lsolid, symmetries=[])

# init = 1 # almost uniformly 1
# m = Blob(n, n; init, lvoid, lsolid, symmetries=[1, 2])

# generate a sample
sharpness = 0.995
a = m(sharpness)
display(heatmap(a))
# error("stop here")

opt = AreaChangeOptimiser(m; maxchange=0.1)
# opt=Adam(1)
opt_state = Flux.setup(opt, m)
for i = 1:40
    l, (dldm,) = Flux.withgradient(m) do m
        circ = [norm([x, y] - [n, n] / 2) < n / 4 for x = 1:n, y = 1:n]
        Flux.mae(circ, m())
    end
    println("($i)")
    println("loss: $l")
    Flux.update!(opt_state, m, dldm)
    println()
end
heatmap(m())

