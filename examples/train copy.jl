using Random, Flux, LinearAlgebra, Porcupine, CUDA
using Flux: gradient, withgradient
using Porcupine: fmap
using AbbreviatedStackTraces
include("../src/main.jl")
gpu(F, x) = cu(F.(x))
cpu(F, x) = Array(F.(x))
cpu(x::Number) = x
gpu(x::Number) = x
cpu(x::AbstractArray{<:Number}) = Array(x)
gpu(x::AbstractArray{<:Number}) = cu(x)
gpu(x::AbstractArray) = gpu.(x)
cpu(x::AbstractArray) = cpu.(x)


gpu(x) = isempty(propertynames(x)) ? x : fmap(gpu, x)
cpu(x) = isempty(propertynames(x)) ? x : fmap(cpu, x)
gpu(d::Dictlike) = fmap(gpu, d)
cpu(d::Dictlike) = fmap(cpu, d)

m = Blob(4, 4; init=1, lmin=1) |> gpu
g = gradient(m) do m
    sum(m())
end