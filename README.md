# Jello.jl

This package is undergoing rapid changes - docs may not be up to date.

## Manufacturable geometry generation for topology optimization & generative design
We design a differentiable Fourier domain algorithm for generating manufacturable geometry in topology optimization & generative inverse design. We approximately bound length scales in any dimension by deriving real space geometry from a Fourier k-space of spatial frequencies via the inverse Fourier transform. This reduces undesirable thin features, close spacings and tight bends that hamper manufacturability or induce checkerboard instability.  We use an adjustable step nonlinearity to induce stable bounded adjoint gradients without an extraneous non-binary density penalty function.

## Adjoint optimization
In real applications, our geometry generator would interface with a FEM or FDM solver that computes a loss function against a target metric. For gradient based adjoint optimization, the solver needs to be amenable to automatic differentiation or have hard coded adjoints. For the sake of testing `Jello.jl`, we pretend we know the optimal geometry (eg circle) and verify that `Jello.jl` can reach it through gradient descent.
```julia
using Test, Random, Flux, CairoMakie, LinearAlgebra, StatsBase, Optim
using Flux: gradient, withgradient
using Optim: Options, minimizer
using Jello

Random.seed!(1)

l = 32
y = float.([norm([x, y] - [l, l] / 2) < l / 4 for x = 1:l, y = 1:l]) # circle
contrast = 10f0
nbasis=4
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
```
![](pic.png)
## Collaboration
LMK if you try it out on an adjoint FEM solver. We can also interface with solvers in C or other languages by passing the Jacobian.
## Contributing
Consider supporting on [Patreon](https://patreon.com/pxshen?utm_medium=clipboard_copy&utm_source=copyLink&utm_campaign=creatorshare_creator&utm_content=join_link) if you found this repo helpful. Feel free to request features or contribute PRs :)
