using Test, Random, Flux, GLMakie, LinearAlgebra, StatsBase
using Flux: gradient, withgradient
using AbbreviatedStackTraces
# using Jello
include("../src/main.jl")

Random.seed!(1)
l = 50
y = float.([norm([x, y] - [l, l] / 2) < l / 4 for x = 1:l, y = 1:l]) # circle
model = Blob(l, l; nbasis=4, contrast=20,)
# model = Blob(l, l; nbasis=4, contrast=10, rmin=3)
iterations = 100

fig = Figure()
empty!(fig)
aspect = 1
heatmap(fig[1, 1], model(), axis=(; aspect, title="start of training"))

loss(model) = mean(abs, y - model())

# train
opt = Adam(0.5)
opt_state = Flux.setup(opt, model)
for i = 1:iterations
    l, (dldm,) = withgradient(loss, model)
    Flux.update!(opt_state, model, dldm)
    i % 10 == 0 && println("$i $l")
end

heatmap(fig[2, 1], model(), axis=(; title="Flux.Adam $iterations steps", aspect))
save("pic.png", fig)
display(fig)