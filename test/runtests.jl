using Test, Random, Flux, CairoMakie, LinearAlgebra, StatsBase, Optim
using Flux: gradient, withgradient
using Optim: Options, minimizer
include("../src/Jello.jl")
using .Jello
# include("../src/mask.jl")

Random.seed!(1)

l = 32
y = float.([norm([x, y] - [l, l] / 2) < l / 4 for x = 1:l, y = 1:l]) # circle
contrast = 10.0f0
nbasis = 4
model = Mask((l, l), nbasis, contrast)
# iterations = 80

fig = Figure()
empty!(fig)
aspect = 1
heatmap(fig[1, 1], model(), axis=(; aspect, title="start of training"))

loss(model) = mean(abs, y - model())

"Optim.jl train"
x0, re = destructure(model)
f = loss ∘ re
function g!(storage, x)
    model = re(x)
    g, = gradient(loss, model)
    storage .= realvec(g.a)
end
function fg!(storage, x)
    model = re(x)
    l, (g,) = withgradient(loss, model)
    storage .= realvec(g.a)
    l
end

od = OnceDifferentiable(f, g!, fg!, x0)
n1 = 20
@showtime res = optimize(od, x0, LBFGS(), Optim.Options(f_tol=0, iterations=n1, show_every=1, show_trace=true))
model1 = re(minimizer(res))

# @showtime res = optimize(od, x0, ParticleSwarm(;
#         n_particles=16), Optim.Options(f_tol=0, iterations=20, show_every=1, show_trace=true))
# model2 = re(minimizer(res))

"Flux.jl train"
opt = Adam(0.5)
opt_state = Flux.setup(opt, model)
n2 = 80
for i = 1:n2
    @time begin
        l, (dldm,) = withgradient(loss, model)
        Flux.update!(opt_state, model, dldm)
        println("$i $l")
    end
end

heatmap(fig[2, 1], model1(), axis=(; title="Optim.LBFGS $n1 steps", aspect))
heatmap(fig[2, 2], model(), axis=(; title="Flux.Adam $n2 steps", aspect))
# heatmap(fig[2, 2], model2(), axis=(; title="ParticleSwarm end of training"))
model = Mask(model1; dims=2 .* model1.dims)
heatmap(fig[3, 2], model(), axis=(; title="resized", aspect))
display(fig)