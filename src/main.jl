using FFTW, UnPack, ArrayPadding, LinearAlgebra, Statistics, SparseArrays, Porcupine, ImageMorphology, Optimisers, Functors, Flux, ChainRulesCore
using Porcupine: keys, values, trim, round, floor, ceil
using Zygote: Buffer, ignore_derivatives, @ignore_derivatives
include("utils.jl")
include("loss.jl")
# include("convblob.jl")
include("interpblob.jl")
include("fourierblob.jl")
include("blob.jl")
include("opt.jl")
# include("sgopt.jl")

# m = Blob(4, 4)
# using Pkg
# pkg"dev C:\Users\pxshe\OneDrive\Desktop\beans\Porcupine.jl;dev C:\Users\pxshe\OneDrive\Desktop\beans\ArrayPadding.jl; up"

# using CUDA, Zygote
# GC.gc()
# @time a = CUDA.zeros(1000, 1000)
# @time b = Zygote.Buffer(a)
# @time b[:] .= 0
# @time diff(a, dims=1)
# 0
