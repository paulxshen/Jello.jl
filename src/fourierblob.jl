struct FourierBlob
    ar::AbstractArray
    ai::AbstractArray
    contrast
    sz
    ose
    cse
    symmetries
    diagonal_symmetry
end
@functor FourierBlob (ar, ai)
Base.size(m::FourierBlob) = m.sz

"""
    FourierBlob(sz...; nbasis=4, contrast=1, T=Float32, rmin=nothing, rminfill=rmin, rminvoid=rmin, symmetries=[], diagonal_symmetry=false)
    (m::FourierBlob)()

Functor for generating length scale controlled geometry mask, represented as array of values between 0 to 1.0. `FourierBlob` constructor makes the callable functor which can be passed as the model parameter to `Flux.jl`. Caliing it yields geometry whose length, spacing and radii are roughly on order of `edge length / nbasis`. contrast controls the edge sharpness. Setting `rmin` applies additional morphological filtering which eliminates smaller features and radii

Args
- `sz`: size of mask array
- `contrast`: edge sharpness
- `nbasis`: # of Fourier basis along each dimension
- `rmin`: minimal radii during morphological filtering, can also be `nothing` (no filtering) or `:auto` (automatically set wrt `nbasis`)
- `rminfill`: same as `rmin` but only applied to fill (bright) features
- `rminvoid`: ditto
"""
function FourierBlob(sz...; nbasis=4, init=nothing, contrast=1, T=Float32, rmin=nothing, rminfill=rmin, rminvoid=rmin, symmetries=[], diagonal_symmetry=false, verbose=true)
    if length(nbasis) == 1
        nbasis = round.(Int, nbasis ./ minimum(sz) .* sz)
    end
    d = length(sz)
    # a = complex.(randn(T, nbasis...), randn(T, nbasis...))
    if isnothing(init)
        ar = randn(T, nbasis...)
        ai = randn(T, nbasis...)
    else
        ar = zeros(T, nbasis...)
        ai = zeros(T, nbasis...)
        if init == 1
            ar[1] = 1
        end
    end

    ose = cse = nothing
    v = minimum(round.(Int, sz ./ nbasis ./ 4))
    rminfill == :auto && (rminfill = v)
    rminvoid == :auto && (rminvoid = v)

    w = "rmin too high relative to nbasis. much of generated geometry may get erased by morphological filtering . consider setting to `:auto` which in this case evaluates to $v"

    if !isnothing(rminfill,)
        if rminfill > v
            @warn w
        end
        ose = circle(rminfill, d) |> centered
    end
    if !isnothing(rminvoid)
        if rminvoid > v
            @warn w
        end
        cse = circle(rminvoid, d) |> centered
    end

    verbose && @info """
     FourierBlob configs
     
     Geometry generation 
     - Fourier k-space size (# of Fourier basis per dimension): $nbasis
     - edge contrast : $contrast

     Morphological filtering (skipped if nothing )
     - min fill radii: $rminfill
     - min void radii: $rminvoid

     Symmetries: $symmetries

     Suppress this message by verbose=false
     Jello.jl is created by Paul Shen <pxshen@alumni.stanford.edu>
     """

    FourierBlob(ar, ai, T(contrast), sz, ose, cse, symmetries, diagonal_symmetry)
end
FourierBlob(sz::Tuple; kw...) = FourierBlob(sz...; kw...)

function (m::FourierBlob)(contrast=m.contrast, σ=x -> 1 / (1 + exp(-x)))
    @unpack ar, ai, sz, ose, cse, symmetries, diagonal_symmetry = m
    a = complex.(ar, ai)
    margins = round.(Int, sz ./ size(a) .* 0.75)
    i = range.(margins .+ 1, margins .+ sz)
    r = real(ifft(pad(a, 0, fill(0, ndims(a)), sz .+ 2 .* margins .- size(a))))[i...]

    if !isempty(symmetries)
        r += reverse(r, dims=symmetries)
        r /= 2
    elseif diagonal_symmetry == true
        r = (r + r') / 2 # diagonal symmetry in this Ceviche challenge

    end
    if !isinf(contrast)
        # r *= 1contrast / mean(abs.(r))
        r *= 1contrast / maximum(abs.(r))

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


function FourierBlob(m::FourierBlob, sz...; contrast=m.contrast,)
    FourierBlob(m.ar, m.ai, contrast, sz, m.ose, m.cse, m.symmetries, m.diagonal_symmetry)
end
