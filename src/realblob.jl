struct RealBlob
    a::AbstractArray
    nn
    contrast::Real
    sz
    ose
    cse
    symmetries
    diagonal_symmetry
end
@functor RealBlob (a,)
Base.size(m::RealBlob) = m.sz

"""
    RealBlob(sz...; nbasis=4, contrast=1, T=Float32, rmin=nothing, rminfill=rmin, rminvoid=rmin, symmetries=[], diagonal_symmetry=false)
    (m::RealBlob)()

Functor for generating length scale controlled geometry mask, represented as array of values between 0 to 1.0. `RealBlob` constructor makes the callable functor which can be passed as the model parameter to `Flux.jl`. Caliing it yields geometry whose length, spacing and radii are roughly on order of `edge length / nbasis`. contrast controls the edge sharpness. Setting `rmin` applies additional morphological filtering which eliminates smaller features and radii

Args
- `sz`: size of mask array
- `contrast`: edge sharpness
- `nbasis`: # of Fourier basis along each dimension
- `rmin`: minimal radii during morphological filtering, can also be `nothing` (no filtering) or `:auto` (automatically set wrt `nbasis`)
- `rminfill`: same as `rmin` but only applied to fill (bright) features
- `rminvoid`: ditto
"""
function RealBlob(sz...;
    nbasis=4, init=nothing, contrast=1, T=Float32,
    rmin=nothing, rminfill=rmin, rminvoid=rmin,
    symmetries=[], diagonal_symmetry=false,
    verbose=true)
    d = length(sz)
    if length(nbasis) == 1
        nbasis = round.(Int, nbasis ./ minimum(sz) .* sz)
    end

    if isnothing(init)
        a = randn(T, nbasis...)
    else
        a = init
    end
    nn = map(CartesianIndices(Tuple(sz))) do i
        i = collect(T.(1 .+ (Tuple(i) .- 1) .* (size(a) .- 1) ./ (sz .- 1)))
        nn = Base.product(range.(floor.(Int, i), ceil.(Int, i))...)

        w = map(nn) do k
            prod(1 .- abs.(i .- k))
        end
        nn, w
    end
    ose, cse = se(rminfill, rminvoid)

    verbose && @info """
     RealBlob configs
     
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

    RealBlob(a, nn, T(contrast), sz, ose, cse, symmetries, diagonal_symmetry)
end
RealBlob(sz::Tuple; kw...) = RealBlob(sz...; kw...)

function (m::RealBlob)()
    @unpack a, contrast, nn, sz, ose, cse, symmetries, diagonal_symmetry = m
    # itp = interpolate(a, BSpline(Linear()))
    # a = σ(a)
    r = map(nn) do (k, w)
        sum(zip(k, w)) do (k, w)
            a[k...] * w
        end
    end
    # r = Buffer(a, sz)
    # imresize!(r, a)
    # r = copy(r)
    r = apply(symmetries, r)
    # r = σ(contrast * r)
    r = apply(σ, contrast, r)
    r = apply(ose, cse, r)
end


# function RealBlob(m::RealBlob, sz...; contrast=m.contrast,)
#     RealBlob(m.ar, m.ai, contrast, sz, m.ose, m.cse, m.symmetries, m.diagonal_symmetry)
# end
