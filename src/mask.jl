using Random, FFTW, Flux, Zygote, UnPack, StatsBase, ArrayPadding, LinearAlgebra

using Flux: @functor, params


struct Mask
    a::AbstractArray
    m
    dims
end
Flux.@functor Mask
Flux.params(m::Mask) = m.a

"""
    Mask(sz, lmin)
    (m::Mask)(α=0.1, symmetries=[]])

Functor for generating length scale controlled geometry mask, represented as array of values between 0 to 1.0. `Mask` constructor makes the callable functor which can be passed as the model parameter to `Flux.jl`. Caliing it yields geometry whose length, spacing and radii are at least `lmin`. α controls the opacity and gradient present. `α = 0` yields a binary morphology without any gradient for adjoint optimization. `α = 0.1` yields an almost black-white mask with some grayscale gradient .

- sz: size of mask array
- lmin: minimum length scale 
- α: opacity
"""
function Mask(dims, lmin)
    d = length(dims)
    t = round.(Int, dims ./ 2 ./ lmin)
    kmax = prod(t)^(1 / d)
    a = complex.(randn(t), randn(t))
    m = [norm(collect(v)) < kmax + 0.1 for v = Iterators.product(axes(a)...)]
    Mask(a, m, dims)
end
f(x, α) = x > 0.0 ? (1 - α) + α * x : α + α * x
function (m::Mask)(α=0.1, symmetries=nothing)
    @unpack a, dims = m
    a = m.m .* a
    k = size(a, 1)
    r = real(ifft(pad(a, 0, fill(0, k), dims .- size(a))))
    r /= 2mean(abs.(r))
    r = f.(r, α)

    if !isnothing(symmetries)
        r += reverse(r, dims=symmetries)
        r /= 2
    end
    r
end
