using Random, FFTW, UnPack, ArrayPadding, LinearAlgebra, Statistics, SparseArrays, Flux, Porcupine, Images
using Porcupine: keys, values, trim, round, floor, ceil
using Zygote: Buffer, ignore_derivatives, @ignore_derivatives
using Flux: @functor, trainable
include("utils.jl")
include("loss.jl")
include("interpblob.jl")
include("fourierblob.jl")
include("blob.jl")
# m = Blob(4, 4)
# using Pkg
# pkg"dev C:\Users\pxshe\OneDrive\Desktop\Porcupine.jl;dev C:\Users\pxshe\OneDrive\Desktop\ArrayPadding.jl; up"
