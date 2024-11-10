include("main.jl")
m = Blob((9, 9), lvoid=15, lsolid=15)
a, l = m(nothing, withloss=true)

# using Pkg
# pkg"dev C:\Users\pxshe\OneDrive\Desktop\Porcupine.jl;dev C:\Users\pxshe\OneDrive\Desktop\ArrayPadding.jl; "
