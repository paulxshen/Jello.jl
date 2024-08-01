using Random, CairoMakie, LinearAlgebra
# using Jello
include("../src/main.jl")

fig2d = Figure()
l = 50

Random.seed!(1)
alg = :interpolation
lmin = 10
contrast = 1
rmin = lmin / 4
m = Blob(l, l; alg, lmin, contrast)
heatmap(fig2d[1, 1], m(); axis=(; title="$l x $l\nalg = :$alg\ncontrast = $contrast\nrmin = $rmin"))
fig2d |> display

# Random.seed!(1)
# alg = :fourier
# nbasis = 4
# contrast = 1
# rmin = nothing
# m = Blob(l, l; alg, nbasis, contrast)
# heatmap(fig2d[1, 2], m(); axis=(; title="$l x $l\nalg = :$alg\nnbasis = $nbasis\ncontrast = $contrast\nrmin = $rmin"))

# Random.seed!(1)
# alg = :interpolation
# nbasis = 8
# m = Blob(l, l; nbasis, contrast)
# heatmap(fig2d[2, 1], m(); axis=(; title="$l x $l\nalg = :$alg\nnbasis = $nbasis\ncontrast = $contrast\nrmin = $rmin"))

# Random.seed!(1)
# contrast = 20
# m = Blob(l, l; nbasis, contrast)
# heatmap(fig2d[2, 2], m(); axis=(; title="$l x $l\nalg = :$alg\nnbasis = $nbasis\ncontrast = $contrast\nrmin = $rmin"))

# # Random.seed!(1)
# # rmin = :auto
# # m = Blob(l, l; nbasis, contrast, rmin)
# # heatmap(fig2d[2, 3], m(); axis=(; title="$l x $l\nalg = :$alg\nnbasis = $nbasis\ncontrast = $contrast\nrmin = :$rmin"))

# Random.seed!(1)
# rmin = 3
# m = Blob(l, l; nbasis, contrast, rmin)
# heatmap(fig2d[2, 3], m(); axis=(; title="$l x $l\nalg = :$alg\nnbasis = $nbasis\ncontrast = $contrast\nrmin = $rmin"))

# save("samples2d.png", fig2d)
# fig2d

# Random.seed!(1)
# l = 40
# nbasis = 4
# contrast = 20
# m = Blob(l, l, l; nbasis, contrast,)
# fig3d = volume(m(); algorithm=:absorption, axis=(; type=Axis3, title="$l x $l x $l, alg = :$alg, nbasis = $nbasis, contrast = $contrast, rmin = $rmin"))
# save("samples3d.png", fig3d)
# fig3d