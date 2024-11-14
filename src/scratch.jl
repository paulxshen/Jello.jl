include("main.jl")
m = Blob((9, 9), lvoid=25, lsolid=25)
a, l, s, v = m(nothing, withloss=true)

# using Pkg
# pkg"dev C:\Users\pxshe\OneDrive\Desktop\Porcupine.jl;dev C:\Users\pxshe\OneDrive\Desktop\ArrayPadding.jl; "
