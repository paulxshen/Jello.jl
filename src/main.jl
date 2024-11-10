using Random, FFTW, UnPack, ArrayPadding, LinearAlgebra, Statistics, SparseArrays, Flux, Functors, Porcupine, ChainRulesCore, Images
using Porcupine: keys, values, trim
using Zygote: Buffer
include("utils.jl")
include("loss.jl")
# include("convblob.jl")
include("interpblob.jl")
# include("fourierblob.jl")
include("blob.jl")
# m = Blob(4, 4)
# g = gradient(m) do m
#     sum(m())
# end
# g[m]

# using Pkg
# pkg"dev C:\Users\pxshe\OneDrive\Desktop\Porcupine.jl;dev C:\Users\pxshe\OneDrive\Desktop\ArrayPadding.jl; up"
