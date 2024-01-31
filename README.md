# Jello.jl

This package is undergoing rapid changes - docs may not be up to date.

## Manufacturable geometry generation for topology optimization & generative design
We design a differentiable Fourier domain algorithm for generating manufacturable geometry in topology optimization & generative inverse design. We approximately bound length scales in any dimension by deriving real space geometry from a Fourier k-space of spatial frequencies via the inverse Fourier transform. This reduces undesirable thin features, close spacings and tight bends that hamper manufacturability or induce checkerboard instability.  We use an adjustable step nonlinearity to induce stable bounded adjoint gradients without an extraneous non-binary density penalty function.

## Adjoint optimization
In real applications, our geometry generator would interface with a FEM or FDM solver that computes a loss function against a target metric. For gradient based adjoint optimization, the solver needs to be amenable to automatic differentiation or have hard coded adjoints. For the sake of testing `Jello.jl`, we pretend we know the optimal geometry (eg circle) and verify that `Jello.jl` can reach it through gradient descent.
```julia
using Test, Random, Flux, GLMakie, LinearAlgebra, StatsBase
using Flux: gradient, withgradient
using Jello

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
```
![](pic.png)
## Collaboration
LMK if you try it out on an adjoint FEM solver. We can also interface with solvers in C or other languages by passing the Jacobian.
## Contributing
Consider supporting on [Patreon](https://patreon.com/pxshen?utm_medium=clipboard_copy&utm_source=copyLink&utm_campaign=creatorshare_creator&utm_content=join_link) if you found this repo helpful. Feel free to request features or contribute PRs :)
