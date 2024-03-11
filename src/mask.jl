function circle(r, d)
    r = round(Int, r)
    [norm(v) <= r for v = Base.product(fill(-r:r, d)...)] # circle
end
struct Blob
    ar::AbstractArray
    ai::AbstractArray
    contrast
    sz
    ose
    cse
    symmetries
    diagonal_symmetry
end
@functor Blob (ar, ai)
Base.size(m::Blob) = m.sz

"""
    Blob(sz...; nbasis=4, contrast=1, T=Float32, rmin=nothing, rminfill=rmin, rminvoid=rmin, symmetries=[], diagonal_symmetry=false)
    (m::Blob)()

Functor for generating length scale controlled geometry mask, represented as array of values between 0 to 1.0. `Blob` constructor makes the callable functor which can be passed as the model parameter to `Flux.jl`. Caliing it yields geometry whose length, spacing and radii are roughly on order of `edge length / nbasis`. contrast controls the edge sharpness. Setting `rmin` applies additional morphological filtering which eliminates smaller features and radii

Args
- `sz`: size of mask array
- `contrast`: edge sharpness
- `nbasis`: # of Fourier basis along each dimension
- `rmin`: minimal radii during morphological filtering
- `rminfill`: same as `rmin` but only applied to fill (bright) features
- `rminvoid`: ditto
"""
function Blob(sz...; nbasis=4, contrast=1, T=Float32, rmin=nothing, rminfill=rmin, rminvoid=rmin, symmetries=[], diagonal_symmetry=false)
    if length(nbasis) == 1
        nbasis = round.(Int, nbasis ./ minimum(sz) .* sz)
    end
    d = length(sz)
    # a = complex.(randn(T, nbasis...), randn(T, nbasis...))
    ar = randn(T, nbasis...)
    ai = randn(T, nbasis...)

    ose = cse = nothing
    if !isnothing(rminfill,)
        ose = circle(rminfill, d) |> centered
    end
    if !isnothing(rminvoid)
        cse = circle(rminvoid, d) |> centered
    end
    Blob(ar, ai, T(contrast), sz, ose, cse, symmetries, diagonal_symmetry)
end

function (m::Blob)(contrast=m.contrast, σ=x -> 1 / (1 + exp(-x)))
    @unpack ar, ai, sz, ose, cse, symmetries, diagonal_symmetry = m
    a = complex.(ar, ai)
    r = real(ifft(pad(a, 0, fill(0, ndims(a)), sz .- size(a))))

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
        r = r .> 0.5
    end

    isnothing(ose) && isnothing(cse) && return r
    m = m0 = 0
    ignore_derivatives() do
        m = r .> 0.5
        m0 = m
        if !isnothing(ose)
            m = opening(m, ose)
        end
        if !isnothing(cse)
            m = closing(m, cse)
        end
        # if !isnothing(ose)
        #     mo = opening(m, ose)
        # else
        #     mo = m
        # end
        # if !isnothing(cse)
        #     mc = closing(m, cse)
        # else
        #     mc = m
        # end
    end

    # f = Figure()
    # display(heatmap(f[1, 1], m))
    # display(heatmap(f[1, 2], mo))
    # display(heatmap(f[1, 3], mc))
    # display(f)
    # error()

    # m1 = mo .|| .!mc
    # r .* m1 + .!(m) .* (.!(m1))
    r .* (m .== m0) + (m - m0 .> 0)
end


function Blob(m::Blob, sz...; contrast=m.contrast,)
    Blob(m.ar, m.ai, contrast, sz, m.ose, m.cse, m.symmetries, m.diagonal_symmetry)
end
