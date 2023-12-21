using Random, FFTW, Flux, UnPack, StatsBase, ArrayPadding, LinearAlgebra

using Flux: @functor, params
F = Float32

struct Mask
    a::AbstractArray
    p
    bounds
    sz
    symmetries
    diagonal_symmetry
end
Flux.@functor Mask
trainable_params = (:a,)
Flux.trainable(m::Mask) = (; a=m.a,)# p=m.p)
# Flux.params(m::Mask) = (; a=m.a,)# p=m.p)

"""
    Mask(sz, lmin; T=Float32, symmetries=[], diagonal_symmetry=false)
    (m::Mask)(contrast=0.1)

Functor for generating length scale controlled geometry mask, represented as array of values between 0 to 1.0. `Mask` constructor makes the callable functor which can be passed as the model parameter to `Flux.jl`. Caliing it yields geometry whose length, spacing and radii are at least `lmin`. contrast controls the opacity and gradient present. `contrast = 0` yields a binary morphology without any gradient for adjoint optimization. `contrast = 0.1` yields an almost black-white mask with some grayscale gradient .

- sz: size of mask array
- lmin: minimum length scale 
- contrast: opacity
"""
function Mask(sz, nbasis, contrast=0.1f0, bounds=(0, 1); T=Float32, symmetries=[], diagonal_symmetry=false)
    if length(nbasis) == 1
        nbasis = round.(Int, nbasis ./ minimum(sz) .* sz)
    end
    a = complex.(randn(T, nbasis...), randn(T, nbasis...))
    # m = [norm(collect(v)) < kmax + 0.1 for v = Iterators.product(axes(a)...)]
    Mask(a, [contrast], bounds, sz, symmetries, diagonal_symmetry)
end
function Mask(sz; lmin::Real, kw...)
    nbasis = round.(Int, sz ./ lmin)
    Mask(sz, ; nbasis, kw...)
end
# f(x::T, contrast) where {T} =
#     x > 0.0 ? (1 - contrast) + contrast * x :
#     contrast + contrast * x
# f(x::T, contrast) where {T} =
#     if x > 0.0
#         x < 1 ? (1 - contrast) + contrast * x : T(1)
#     else
#         -1 < x ? contrast + contrast * x : T(0)
#     end
function (m::Mask)(contrast=m.p[1], σ=x -> Flux.σ(x))
    # function (m::Mask)(contrast=0.1f0, ; σ=x -> f(x, contrast))
    @unpack a, sz, bounds, symmetries, diagonal_symmetry = m
    contrast = min(contrast, bounds[2])
    contrast = max(contrast, bounds[1])
    r = real(ifft(pad(a, 0, fill(0, ndims(a)), sz .- size(a))))

    if !isempty(symmetries)
        r += reverse(r, dims=symmetries)
        r /= 2
    elseif diagonal_symmetry == true
        r = (r + r') / 2 # diagonal symmetry in this Ceviche challenge

    end
    if !isinf(contrast)
        r *= 10contrast / mean(abs.(r))

        r = σ.(r)
    else
        # σ = isnothing(contrast) ? x -> x + 0.5f0 : x -> f(x, contrast)
        # r = max.(r, 0.0f0)
        # r = min.(r, 1.0f0)
        return F.(r .> 0)
    end
    r
end

# function resize(m, sz)
#     # a = fft(imresize(m(; σ=identity), sz))[range.(1, m.sz)...]
#     a = fft(imresize(m(nothing), sz))[range.(1, size(m.a))...]
#     Mask(a, sz, m.symmetries, m.diagonal_symmetry)
# end

function cvec(v,)
    a = reshape(v, length(v) ÷ 2, 2)
    complex.(a[:, 1], a[:, 2])
end
realvec(a::AbstractArray{<:Real}) = vec(a)
function realvec(a::AbstractArray{<:Complex})
    v = vec(a)
    vcat(real(v), imag(v))
end
function destructure(m::NamedTuple)
    realvec(vcat([vec(v) for (k, v) = pairs(m) if k in trainable_params]...)), nothing
end
function destructure(m::Mask)
    x, re = Flux.destructure(m)
    # vec(m), re ∘ x -> vcat(cvec(x[1:end-1]), [x[end]])
    realvec(x), re ∘ (x -> cvec(x))
end
function optimfuncs(loss, re)

    function f(x)
        model = re(x)
        loss(model)
    end

    function g!(storage, x)
        model = re(x)
        g = gradient(loss, model)[1]
        storage .= destructure(g)[1]
    end
    function fg!(storage, x)
        model = re(x)
        l, (g,) = withgradient(loss, model)
        storage .= destructure(g)[1]
        # pr
        l
    end
    f, g!, fg!
end