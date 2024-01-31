using Test, Random, Flux, GLMakie, LinearAlgebra, StatsBase
using Flux: gradient, withgradient
include("../src/Jello.jl")
using .Jello

Random.seed!(1)

l = 32
y = float.([norm([x, y] - [l, l] / 2) < l / 4 for x = 1:l, y = 1:l]) # circle
contrast = 10.0f0
nbasis = 4
model = Mask((l, l), nbasis, contrast)
iterations = 80

fig = Figure()
empty!(fig)
aspect = 1
heatmap(fig[1, 1], model(), axis=(; aspect, title="start of training"))

loss(model) = mean(abs, y - model())

"Flux.jl train"
opt = Adam(0.5)
opt_state = Flux.setup(opt, model)
for i = 1:iterations
    l, (dldm,) = withgradient(loss, model)
    Flux.update!(opt_state, model, dldm)
    i % 10 == 0 && println("$i $l")
end

heatmap(fig[2, 1], model(), axis=(; title="Flux.Adam $iterations steps", aspect))
# heatmap(fig[2, 2], model2(), axis=(; title="ParticleSwarm end of training"))
model_ = Mask(model; dims=2 .* model.dims)
heatmap(fig[3, 1], model_(), axis=(; title="resized", aspect))
display(fig)
save("pic.png", fig)