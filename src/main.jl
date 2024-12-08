using Random, FFTW, UnPack, ArrayPadding, LinearAlgebra, Statistics, SparseArrays, Flux, Porcupine, ImageMorphology, Optimisers
using Porcupine: keys, values, trim, round, floor, ceil
using Zygote: Buffer, ignore_derivatives, @ignore_derivatives
using Flux: trainable
include("utils.jl")
include("loss.jl")
include("interpblob.jl")
include("fourierblob.jl")
include("blob.jl")
# m = Blob(4, 4)
# using Pkg
# pkg"dev C:\Users\pxshe\OneDrive\Desktop\beans\Porcupine.jl;dev C:\Users\pxshe\OneDrive\Desktop\beans\ArrayPadding.jl; up"
