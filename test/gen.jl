# using Random, Jello, CairoMakie
using Random, CairoMakie, Flux
include("../src/main.jl")

Random.seed!(1)
n = 100
lvoid = 10
lsolid = 10
init = nothing
# init = 1
m = Blob(n, n; init, lvoid, lsolid, periodic=false, symmetries=[])
# m = gpu(m)
sharpness = 0.99
a = m()

display(heatmap(a))
# error("stop here")

opt = Flux.Adam(0.1)
opt_state = Flux.setup(opt, m)
for i = 1:40
    global l, (dldm,) = Flux.withgradient(m) do m
        a, l1 = m(withloss=true)
        l1 = max(l1, 1.0f-2)
        # l2 = mae([norm([x, y] - [n, n] / 2) < n / 4 for x = 1:n, y = 1:n], a)
        l2 = 0
        println(l1, l2)
        l1 + l2
    end
    Flux.update!(opt_state, m, dldm)# |> gpu)
end

heatmap(m())
# # heatmap(s=m.conv.weight[:, :, 1, 1])

# using Pkg
# pkg"dev C:\Users\pxshe\OneDrive\Desktop\beans\Porcupine.jl;dev C:\Users\pxshe\OneDrive\Desktop\beans\ArrayPadding.jl; up"