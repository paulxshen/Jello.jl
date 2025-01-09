struct FourierBlob
    p::AbstractArray
    sz
    sesolid
    sevoid
    symmetries
end
Base.size(m::FourierBlob) = m.sz

"""
    FourierBlob(sz...; nbasis=4, contrast=1, T=Float32, rmin=nothing, rvalssolid=rmin, rvalsvoid=rmin, symmetries=[], diagonal_symmetry=false)
    (m::FourierBlob)()

Functor for generating length scale controlled geometry mask, represented as array of values between 0 to 1.0. `FourierBlob` constructor makes the callable functor which can be passed as the model parameter to `Flux.jl`. Caliing it yields geometry whose length, spacing and radii are roughly on order of `edge length / nbasis`. contrast controls the edge sharp. Setting `rmin` applies additional morphological filtering which eliminates smaller features and radii

Args
- `sz`: size of mask array
- `contrast`: edge sharp
- `nbasis`: # of Fourier basis along each dimension
- `rmin`: minimal radii during morphological filtering, can also be `nothing` (no filtering) or `:auto` (automatically set wrt `nbasis`)
- `rvalssolid`: same as `rmin` but only applied to fill (bright) features
- `rvalsvoid`: ditto
"""
# function FourierBlob(sz...; nbasis=4, solid_frac=nothing, contrast=1, T=Float32, rmin=nothing, rvalssolid=rmin, rvalsvoid=rmin, symmetries=[], diagonal_symmetry=false, verbose=true)
#     if length(nbasis) == 1
#         nbasis = round.(Int, nbasis ./ minimum(sz) .* sz)
#     end
#     d = length(sz)
#     # a = complex.(randn(T, nbasis...), randn(T, nbasis...))

# end

function (m::FourierBlob)(sharp=true)
    @unpack p, sz, sesolid, sevoid, symmetries, = m
    T = eltype(p)
    α = T(1 - sharp) / 2
    pre, pim = eachslice(p, dims=ndims(p))
    p = complex.(pre, pim)
    # margins = round.(Int, sz ./ size(a) .* 0.75)
    # i = range.(margins .+ 1, margins .+ sz)
    # r = real(ifft(pad(a, 0, fill(0, ndims(a)), sz .+ 2 .* margins .- size(a))))[i...]
    a = real(ifft(pad(p, 0, 0, sz .- size(p))))

    a = apply_symmetries(a, symmetries)
    if sharp
        a = stepfunc(a)
    end
    # if sharp > 0
    #     a = smooth(a, sesolid, sevoid)
    # end
    a
end

