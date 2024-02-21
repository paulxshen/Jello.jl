using Random, FFTW, UnPack, StatsBase, ArrayPadding, LinearAlgebra, Functors

# F = Float32

struct Mask
    a::AbstractArray
    contrast
    dims
    symmetries
    diagonal_symmetry
end
@functor Mask (a,)
"""
    Mask(dims, nbasis, contrast=0.2f0; T=Float32, symmetries=[], diagonal_symmetry=false)
    Mask(dims, contrast=0.2f0; lmin::Real, kw...)
    (m::Mask)()

Functor for generating length scale controlled geometry mask, represented as array of values between 0 to 1.0. `Mask` constructor makes the callable functor which can be passed as the model parameter to `Flux.jl`. Caliing it yields geometry whose length, spacing and radii are roughly at least `lmin ≈ dims / nbasis`. contrast controls the edge sharpness. High contrast >1 yields almost a binary image which limits amount of adjoint gradient present.  `contrast = 0.2` yields a smooth mask slightly sharper than the Fourier basis output and is a good starting point for adjoint optimization.

Args
- `dims`: size of mask array
- `lmin`: minimum length scale 
- `contrast`: edge sharpness
- `nbasis`: # of Fourier basis along each dimension
"""
function Mask(dims, nbasis, contrast=0.2f0; T=Float32, symmetries=[], diagonal_symmetry=false)
    if length(nbasis) == 1
        nbasis = round.(Int, nbasis ./ minimum(dims) .* dims)
    end
    a = complex.(randn(T, nbasis...), randn(T, nbasis...))
    Mask(a, T(contrast), dims, symmetries, diagonal_symmetry)
end
function Mask(dims, a...; lmin::Real, kw...)
    nbasis = round.(Int, dims ./ lmin)
    Mask(dims, nbasis, a...; kw...)
end

function (m::Mask)(contrast=m.contrast, σ=x -> 1 / (1 + exp(-x)))
    @unpack a, dims, symmetries, diagonal_symmetry = m
    # contrast = min(contrast, bounds[2])
    # contrast = max(contrast, bounds[1])
    r = real(ifft(pad(a, 0, fill(0, ndims(a)), dims .- size(a))))

    if !isempty(symmetries)
        r += reverse(r, dims=symmetries)
        r /= 2
    elseif diagonal_symmetry == true
        r = (r + r') / 2 # diagonal symmetry in this Ceviche challenge

    end
    if !isinf(contrast)
        r *= 1contrast / mean(abs.(r))

        r = σ.(r)
    else
        return r .> 0
    end
    r
end


function Mask(m::Mask; dims=m.dims, contrast=m.contrast, params=m.a)
    Mask(params, contrast, dims, m.symmetries, m.diagonal_symmetry)
end


function cvec(v,)
    a = reshape(v, length(v) ÷ 2, 2)
    complex.(a[:, 1], a[:, 2])
end
# realvec(a::AbstractArray{<:Real}) = vec(a)
function realvec(a::AbstractArray{<:Complex})
    v = vec(a)
    vcat(real(v), imag(v))
end
function destructure(m::Mask)
    realvec(m.a), x -> Mask(m; params=reshape(cvec(x), size(m.a)))
end