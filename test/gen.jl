# using Random, Jello, CairoMakie
using Random, CairoMakie, Flux
include("../src/main.jl")

Random.seed!(1)
n = 100
lvoid = 10
lsolid = 10
init = nothing
# init = 1
m = Blob(n, n; init, lvoid, lsolid, periodic=true, symmetries=[1, 2])
# m = gpu(m)
sharpness = 0.99
a = m()

display(heatmap(a))

opt = Flux.Adam(0.1)
opt_state = Flux.setup(opt, m)
for i = 1:10
    global l, (dldm,) = Flux.withgradient(m) do m
        l = mae([norm([x, y] - [n, n] / 2) < n / 4 for x = 1:n, y = 1:n], m())
        println(l)
        l
    end
    Flux.update!(opt_state, m, dldm)# |> gpu)
end

heatmap(m())
# # heatmap(s=m.conv.weight[:, :, 1, 1])