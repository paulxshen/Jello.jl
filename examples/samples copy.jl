using Random, GLMakie, LinearAlgebra
# using Jello
include("../src/main.jl")

fig2d = Figure()
l = 100

Random.seed!(1)
contrast = 1
rmin = nothing
m = RealBlob(l, l; contrast)
# heatmap(fig2d[1, 1], m(); axis=(; title="$l x $l\nnbasis = $nbasis\ncontrast = $contrast\nrmin = $rmin"))

# Random.seed!(1)
# nbasis = 6
# m = RealBlob(l, l; nbasis, contrast)
# heatmap(fig2d[2, 1], m(); axis=(; title="$l x $l\nnbasis = $nbasis\ncontrast = $contrast\nrmin = $rmin"))

# Random.seed!(1)
# contrast = 20
# m = RealBlob(l, l; nbasis, contrast)
# heatmap(fig2d[2, 2], m(); axis=(; title="$l x $l\nnbasis = $nbasis\ncontrast = $contrast\nrmin = $rmin"))

# Random.seed!(1)
# rmin = :auto
# m = RealBlob(l, l; nbasis, contrast, rmin)
# heatmap(fig2d[2, 3], m(); axis=(; title="$l x $l\nnbasis = $nbasis\ncontrast = $contrast\nrmin = :$rmin"))

Random.seed!(1)
rmin = 5
contrast = 10
m = RealBlob(l, l; contrast, rmin)
heatmap(fig2d[2, 4], m(); axis=(; title="$l x $l\ncontrast = $contrast\nrmin = $rmin"))

# save("samples2d.png", fig2d)
fig2d

# Random.seed!(1)
# l = 40
# contrast = 20
# m = RealBlob(l, l, l;  contrast,rmin)
# fig3d = volume(m(); algorithm=:absorption, axis=(; type=Axis3, title="$l x $l x $l, nbasis = $nbasis, contrast = $contrast, rmin = $rmin"))
# save("samples3d.png", fig3d)
# fig3d