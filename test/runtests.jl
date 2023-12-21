using Test, Random, Flux, CairoMakie, LinearAlgebra, Optim
using Flux: gradient, withgradient
using Optim: Options, minimizer
# include("../src/Jello.jl")
# using .Jello
include("../src/mask.jl")

Random.seed!(1)

l = 128
y = float.([norm([x, y] - [l, l] / 2) < l / 4 for x = 1:l, y = 1:l]) # circle
# contrast = .1f0
model = Mask((l, l), 8, 1.0f0)

fig = Figure()
# empty!(fig)
heatmap(fig[1, 1], model(), axis=(; title="start of training"))

loss(model) = mean(abs, y - model())
x0, re = destructure(model)
f, g!, fg! = optimfuncs(loss, re)
od = OnceDifferentiable(f, g!, fg!, x0)
@showtime res = optimize(od, x0, LBFGS(), Optim.Options(f_tol=0, iterations=20, show_every=1, show_trace=true))
# res = 

model = re(minimizer(res))
heatmap(fig[1, 2], model(), axis=(; title="end of training"))
heatmap(fig[2, 1], model(), axis=(; title="raw"))
# model.sz = 2 .* model.sz
# # m = resize(m, 2 .* m.sz)
# heatmap(fig[2, 2], model(), axis=(; title="resized"))
display(fig)