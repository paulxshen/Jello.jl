using Random, FFTW, Flux, UnPack, StatsBase, ArrayPadding, LinearAlgebra

using Flux: @functor, params
F = Float32

struct Mask
    a::AbstractArray
    p
    bounds
    dims
    symmetries
    diagonal_symmetry
end
Flux.@functor Mask
trainable_params = (:a,)
Flux.trainable(m::Mask) = (; a=m.a,)# p=m.p)

struct NNMask
    f
    p
    dims
    symmetries
    diagonal_symmetry
end
Flux.@functor NNMask
# Flux.trainable(m::NNMask) = (; f=m.f)# p=m.p)
# `contrast = 0` yields a binary morphology without any gradient for adjoint optimization.
"""
    Mask(dims, nbasis, contrast=0.2f0, bounds=(0, 1); T=Float32, symmetries=[], diagonal_symmetry=false)
    Mask(dims, contrast=0.2f0, bounds=(0, 1); lmin::Real, kw...)
    (m::Mask)()

Functor for generating length scale controlled geometry mask, represented as array of values between 0 to 1.0. `Mask` constructor makes the callable functor which can be passed as the model parameter to `Flux.jl`. Caliing it yields geometry whose length, spacing and radii are roughly at least `lmin ≈ dims / nbasis`. contrast controls the edge sharpness. High contrast >1 yields almost a binary image which limits amount of adjoint gradient present.  `contrast = 0.2` yields a smooth mask slightly sharper than the Fourier basis output and is a good starting point for adjoint optimization.

Args
- `dims`: size of mask array
- `lmin`: minimum length scale 
- `contrast`: edge sharpness
- `nbasis`: # of Fourier basis along each dimension
"""
function Mask(dims, nbasis, contrast=0.2f0, bounds=(0, 1); T=Float32, symmetries=[], diagonal_symmetry=false)
    if length(nbasis) == 1
        nbasis = round.(Int, nbasis ./ minimum(dims) .* dims)
    end
    a = complex.(randn(T, nbasis...), randn(T, nbasis...))
    Mask(a, [contrast], bounds, dims, symmetries, diagonal_symmetry)
end
function Mask(dims, a...; lmin::Real, kw...)
    nbasis = round.(Int, dims ./ lmin)
    Mask(dims, nbasis, a...; kw...)
end

function (m::Mask)(contrast=m.p[1], σ=Flux.σ)
    @unpack a, dims, bounds, symmetries, diagonal_symmetry = m
    contrast = min(contrast, bounds[2])
    contrast = max(contrast, bounds[1])
    r = real(ifft(pad(a, 0, fill(0, ndims(a)), dims .- size(a))))

    if !isempty(symmetries)
        r += reverse(r, dims=symmetries)
        r /= 2
    elseif diagonal_symmetry == true
        r = (r + r') / 2 # diagonal symmetry in this Ceviche challenge

    end
    if !isinf(contrast)
        r *= 10F(contrast) / mean(abs.(r))

        r = σ.(r)
    else
        return r .> 0
    end
    r
end
function NNMask(dims, f, contrast=0.2f0; T=Float32, symmetries=[], diagonal_symmetry=false)
    NNMask(f, [contrast], dims, symmetries, diagonal_symmetry)
end

function (m::NNMask)(contrast=m.p[1], σ=Flux.σ)
    @unpack f, dims, symmetries, diagonal_symmetry = m
    r = [f(collect(t) ./ dims)[1] for t = Iterators.product(Base.oneto.(dims)...)]

    if !isempty(symmetries)
        r += reverse(r, dims=symmetries)
        r /= 2
    elseif diagonal_symmetry == true
        r = (r + r') / 2 # diagonal symmetry in this Ceviche challenge

    end
    # r *= 10contrast / mean(abs.(r))

    # r = σ.(r)
    r
end

function Mask(m::Mask; dims=m.dims, contrast=m.p[1])
    Mask(m.f, [contrast], dims, m.symmetries, m.diagonal_symmetry)
end
function NNMask(m; dims=m.dims, contrast=m.p[1])
    NNMask(m.f, [contrast], dims, m.symmetries, m.diagonal_symmetry)
end

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
        g = gradient(() -> loss(model), params(model))[1]
        storage .= realvec(g)
    end
    function fg!(storage, x)
        model = re(x)
        l, (g,) = withgradient(() -> loss(model), params(model))
        # println(g)
        storage .= realvec(g)
        l
    end
    f, g!, fg!
end