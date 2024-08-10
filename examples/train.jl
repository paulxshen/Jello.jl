using Random, Flux, LinearAlgebra
using Flux: gradient, withgradient
using AbbreviatedStackTraces
# using Jello
include("../src/main.jl")

Random.seed!(1)
l = 30
y = float.([norm([x, y] - [l, l] / 2) < l / 4 for x = 1:l, y = 1:l]) # circle
model = Blob(l, l; alg=:interpolation, nbasis=8, contrast=20,)
# model = Blob(l, l; alg=:fourier, nbasis=10, contrast=10,)
iterations = 20

fig = Figure()
empty!(fig)
aspect = 1
heatmap(fig[1, 1], model(), axis=(; aspect, title="start of training"))

loss(model) = mean(abs, y - model())

# train
opt = Adam(0.5)
opt_state = Flux.setup(opt, model)
for i = 1:iterations
    l, (dldm,) = withgradient(loss, model)
    Flux.update!(opt_state, model, dldm)
    i % 1 == 0 && println("$i $l")
end

heatmap(fig[2, 1], model(), axis=(; title="Flux.Adam $iterations steps", aspect))
save("pic.png", fig)
display(fig)