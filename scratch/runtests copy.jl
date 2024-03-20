using Test, Random, GLMakie, LinearAlgebra, StatsBase, Optim
using Optim: Options, minimizer
using Jello
include("../../startup.jl")

Random.seed!(1)

l = 32
y = float.([norm([x, y] - [l, l] / 2) < l / 4 for x = 1:l, y = 1:l]) # circle
contrast = 10.0f0
nbasis = 4
model = FourierBlob((l, l), nbasis, contrast)
# iterations = 80

fig = Figure()
empty!(fig)
aspect = 1

loss(model) = mean(abs, y - model())

"Optim.jl train"
x0, re = destructure(model)
f = loss ∘ re


@showtime res = optimize(f, x0, ParticleSwarm(n_particles=32), Optim.Options(f_tol=0, iterations=200, show_every=1, show_trace=true))
# @showtime res = optimize(f, x0,NelderMead(), Optim.Options(f_tol=0, iterations=200, show_every=1, show_trace=true))
# model = re(minimizer(res))
heatmap(model(), axis=(; title="Flux.Adam $n2 steps", aspect))


# demo resizing
model_ = FourierBlob(model, (2 .* size(model))...)
heatmap(fig[3, 1], model_(), axis=(; title="resized", aspect))
