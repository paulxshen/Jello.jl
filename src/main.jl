using Random, FFTW, UnPack, ArrayPadding, LinearAlgebra, Statistics, SparseArrays, Flux, Zygote, Functors, ImageMorphology, Porcupine, ChainRulesCore, NNlib, ImageTransformations
# using ImageTransformations
# using Zygote: Buffer
include("utils.jl")
include("convblob.jl")
include("interpblob.jl")
include("fourierblob.jl")
include("blob.jl")
# m = Blob(4, 4)
# g = gradient(m) do m
#     sum(m())
# end
# g[m]
# g = gradient(Params([m.a])) do
#     sum(m())
# end