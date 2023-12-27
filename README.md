# Jello.jl

## Manufacturable geometry generation for topology optimization & generative design
We design a differentiable Fourier domain algorithm for generating manufacturable geometry in topology optimization & generative inverse design. We approximately bound length scales in any dimension by deriving real space geometry from a Fourier k-space of spatial frequencies via the inverse Fourier transform. This reduces undesirable thin features, close spacings and tight bends that hamper manufacturability or induce checkerboard instability.  We use an adjustable step nonlinearity to induce stable bounded adjoint gradients without an extraneous non-binary density penalty function.

## Usage
<!-- ## Usage -->
```julia
using Jello, Random
Random.seed!(1)

l = 128
dims = (l, l)
lmin = 16
m = Mask(dims, lmin)

contrast = 0.1
a = m(contrast)
```
![](sample.png)
## Adjoint optimization
In real applications, our geometry generator would interface with a FEM or FDM solver that computes a loss function against a target metric. For gradient based adjoint optimization, the solver needs to be amenable to automatic differentiation or have hard coded adjoints. For the sake of testing `Jello.jl`, we pretend we know the optimal geometry (eg circle) and verify that `Jello.jl` can reach it through gradient descent.
```julia

```
![](train.png)
## Collaboration
LMK if you try it out on an adjoint FEM solver. We can also interface with solvers in C or other languages by passing the Jacobian.
## Contributing
Consider supporting on [Patreon](https://patreon.com/pxshen?utm_medium=clipboard_copy&utm_source=copyLink&utm_campaign=creatorshare_creator&utm_content=join_link) if you found this repo helpful. Feel free to request features or contribute PRs :)
