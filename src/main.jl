using Random, FFTW, UnPack, ArrayPadding, LinearAlgebra, Statistics, Flux, Zygote, Functors, ImageMorphology, Porcupine, ChainRulesCore, NNlib, ImageTransformations
# using ImageTransformations
# using Zygote: Buffer
include("utils.jl")
include("realblob.jl")
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