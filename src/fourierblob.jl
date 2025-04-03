struct FourierBlob
    p::AbstractArray
    sz
    asz
    mask
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

function (m::FourierBlob)()
    @unpack p, sz, asz, mask, symmetries, = m
    pre, pim = eachslice(p, dims=ndims(p))
    p = complex.(pre, pim) .* mask
    a = real(ifft(pad(p, 0, 0, asz - size(p))))
    margins = (asz - sz) .÷ 2
    a = a[range.(margins + 1, margins + sz)...]
    a = apply_symmetries(a, symmetries)
    a = stepfunc.(a)
end

