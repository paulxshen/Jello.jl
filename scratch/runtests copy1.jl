using Test, Random, Flux, CairoMakie, LinearAlgebra
include("../src/Jello.jl")
Random.seed!(1)

fig = Figure()
l = 128
dims = (l, l)

for (i, lmin) = enumerate(round.(Int, [l / 8, l / 12]))
    m = Mask(dims, lmin)
    for (j, contrast) = enumerate([0.1, 0.3])
        axis = (; title="$l x $l\nlmin = $lmin\ncontrast = $contrast\n")
        heatmap(fig[i, j], m(contrast); axis)
    end
end
display(fig)

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
@showtime res = optimize(od, x0, LBFGS(), Optim.Options(f_tol=0, iterations=n1, show_every=10, show_trace=true))
println(res)
model1 = re(minimizer(res))

# @showtime res = optimize(od, x0, ParticleSwarm(;
#         n_particles=16), Optim.Options(f_tol=0, iterations=20, show_every=10, show_trace=true))
# model2 = re(minimizer(res))